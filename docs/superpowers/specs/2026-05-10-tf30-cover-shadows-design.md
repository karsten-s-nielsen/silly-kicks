# TF-30: Cover Shadow Features — Lane Control + Blocking Score

**Date:** 2026-05-10
**Status:** Approved
**PR:** PR-S36
**Version target:** silly-kicks 3.11.0

## 1. Summary

New defensive feature category — lane-specific pass obstruction (off-ball defensive
value). Implements the Lane Control physics-based pass-blocking model and blocking
score counterfactual threat reduction metric from Cascioli, Wang, Stradiotti, Van Roy,
Robberechts, Wouters, Jaspers & Davis 2025 (Hudl/DTAI, KU Leuven — same research
group as VAEP).

Three pieces ship:

1. **Lane Control** — per-(passer, receiver) pass-blocking probability via
   corridor-discretized TTI race.
2. **Blocking Score** — per-defender threat reduction via counterfactual removal,
   using xT x pitch control as the threat layer.
3. **Action-coupled layer** — `add_cover_shadows` aggregator (5 columns),
   `cover_shadow_xfns` VAEP factory, atomic mirror.

Out of scope: SoccerMap CNN pass selection/success surfaces (paper's RQ2 threat model
uses these; we substitute xT x pitch control), defensive positioning optimization
(paper's RQ3 — coaching tool, not a per-frame VAEP feature).

## 2. References

- Cascioli, L., Wang, A., Stradiotti, L., Van Roy, M., Robberechts, P., Wouters, M.,
  Jaspers, A., & Davis, J. (2025). "Quantifying Off-Ball Defensive Impact through
  Cover Shadows." Hudl Research / DTAI, KU Leuven.
- Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017). "Physics-Based
  Modeling of Pass Probabilities in Soccer." MIT Sloan SAC. (Ball drag model)
- Bekkers, J. (2024). "Pressing Intensity: An Intuitive Measure for Pressing in
  Soccer." arXiv:2501.04712. (TTI concept extended by Cascioli et al.)

## 3. Architecture

### 3.1 Module Layout

Single new module: `silly_kicks/tracking/_cover_shadows.py` (~400 LOC).

Not a subpackage — one paper, one methodology (unlike pitch control's three-model
dispatch warranting a subpackage per ADR-008).

Action-coupled wrappers live in `silly_kicks/tracking/features.py` and
`silly_kicks/atomic/tracking/features.py` (mechanical mirror).

### 3.2 Public API Surface

```python
# _cover_shadows.py
CoverShadowParams          # frozen dataclass — all tunable constants
LaneControlResult           # frozen dataclass — per-pair blocking probabilities
lane_control()              # per-(passer, receiver) primitive
compute_blocking_score()    # per-frame counterfactual primitive

# features.py additions
add_cover_shadows()         # aggregator -> 5 columns
cover_shadow_xfns()         # VAEP factory -> 15 VAEP columns (5 x 3 states)
```

### 3.3 Dependencies (all shipped)

- `pitch_control/` — `compute_pitch_control` for counterfactual threat model
- `xthreat.py` — `ExpectedThreat` for positional threat grid
- `_gk_resolve.py` — `defending_gk_from_frames` (GK exclusion in receiver set)
- `utils.py` — `link_actions_to_frames` (action-coupled layer)

## 4. Lane Control Primitive

### 4.1 Corridor Parameterization

Pass direction unit vector `u = (r - p) / ||r - p||`, perpendicular
`u_perp = (-u_y, u_x)`, cone half-width at receiver
`w/2 = k x ||r - p|| / 2` (k=0.2, so +/-1m offset per 10m pass length).

Three lines, each 30 evenly-spaced points from passer toward receiver:

```
center(t) = p + t x (r - p)                        t in linspace(0, 1, 30)
left(t)   = center(t) + t x (w/2) x u_perp         (cone expands linearly)
right(t)  = center(t) - t x (w/2) x u_perp
```

At t=0 all three lines converge at the passer. At t=1 they fan out +/-w/2 at the
receiver.

### 4.2 Ball Drag Model (Spearman 2017)

Quadratic drag, no gravity/Magnus:

```
k_drag = (rho x C_D x A) / (2 x m)
       = (1.22 x 0.25 x 0.038) / (2 x 0.42) ~ 0.01383

T_ball(d) = expm1(k_drag x d) / (v0 x k_drag)
```

Default `v0 = 12.0 m/s`. Uses `np.expm1` for numerical stability when `k_drag x d`
is small.

### 4.3 Player TTI (3-phase accelerate-then-cruise)

Per player per target point:

```
r_react = r_player + v_player x t_react          (position after reaction delay)
d = ||q_target - r_react||                        (distance to target)
d_eff = max(d - r_block, 0)  [defenders]          (block radius advantage)
d_eff = d                    [attackers]           (must reach exact point)
e_hat = (q_target - r_react) / d                  (unit toward target)
v0 = max(0, v_player . e_hat)                     (velocity component, clamped >= 0)
t_accel = (v_max - v0) / a_max                    (time to reach max speed)
d_accel = v0 x t_accel + 0.5 x a_max x t_accel^2  (distance during acceleration)
```

Three piecewise cases:

| Case              | Condition          | TTI                                                           |
|-------------------|--------------------|---------------------------------------------------------------|
| Cruising          | v0 >= v_max        | t_react + d_eff / v_max                                      |
| Acceleration only | d_eff <= d_accel   | t_react + (-v0 + sqrt(v0^2 + 2 x a_max x d_eff)) / a_max    |
| Accel + cruise    | d_eff > d_accel    | t_react + t_accel + (d_eff - d_accel) / v_max                |

Vectorized: `pos (n_players, 2)`, `vel (n_players, 2)`, `targets (n_points, 2)` ->
broadcast to `(n_players, n_points)` TTI matrix.

This is a new implementation distinct from the existing Spearman TTI (`compute_tti`)
and Bekkers TTI (`_bekkers_tti`). Key differences: explicit max speed cap, block
radius for defenders, velocity component clamped non-negative. Implements the exact
model from Cascioli et al. 2025.

**Relationship to `compute_tti`:** The new 3-phase model subsumes the Spearman TTI.
Setting `max_speed → ∞` and `block_radius = 0` reduces to the acceleration-only
quadratic formula identical to `compute_tti` (which lacks a max speed cap and block
radius). The paper notes that "some popular physics-based implementations do not
consider the acceleration phase and simply assume that players run at max speed. We
found this to clearly worsen the method performance." Consolidation of the TTI
implementations is a future opportunity but out of TF-30 scope — the Spearman TTI is
tightly coupled to the pitch control model's existing API.

### 4.4 Probability Conversion

Per target point k, per player j:

```
dt_j = T_ball(k) - TTI_j(k)                      (positive = player arrives before ball)
s = sqrt(3) x sigma / pi,  sigma = 0.20  ->  s ~ 0.1103
P_int_j = 1 / (1 + exp(-dt_j / s))               (interception probability)

dT_k = T_ball(k) - T_ball(k-1)                   (time interval between points)
P_ctrl = 1 - exp(-lambda x dT_k),  lambda = 4.3  (ball control probability)

P_j(k) = P_int_j(k) x P_ctrl(k) x (1 - P_anyone_prior(k))
```

`P_anyone_prior(k)` is the running cumulative probability that any player intercepted
at earlier points. Sequential integration runs from passer (k=0) toward receiver (k=29).

Per line, aggregate into `P_blocked = sum(P_defenders)` and
`P_received = sum(P_attackers)`. Line blocked if `P_blocked > P_received`.

### 4.5 Man-Marking Filter

Before lane control, classify each defender:

```
For each defender d, for each attacker a:
    behind_point = a_pos + 1.0 x unit_toward_own_goal
    if ||d_pos - behind_point|| < 3.0m:
        d is man-marking -> exclude from lane-blocking analysis
```

"Unit toward own goal" resolved from `home_team_id` (home defends x=0, away defends
x=105). GK is always excluded from the lane-blocker set.

### 4.6 Return Type

```python
@dataclass(frozen=True)
class LaneControlResult:
    p_blocked_center: float
    p_blocked_left: float
    p_blocked_right: float
    p_received_center: float
    p_received_left: float
    p_received_right: float
    is_blocked_any: bool       # True if ANY line blocked
    is_blocked_majority: bool  # True if >= 2 lines blocked
    is_blocked_all: bool       # True if ALL 3 lines blocked
```

### 4.7 Function Signature

```python
def lane_control(
    frame: pd.DataFrame,
    passer_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    *,
    home_team_id: int | str,
    attacking_team_id: int | str,
    params: CoverShadowParams | None = None,
) -> LaneControlResult:
```

Takes a single frame, passer/receiver positions, returns blocking probabilities and
decision flags. Pure function, no side effects.

**LTR validation:** `lane_control` raises `ValueError` if frames contain non-LTR
direction values, following the established pattern from `_off_ball_runs._validate_ltr()`
and `_defensive_line.compute_defensive_line()`. The man-marking filter's "unit toward
own goal" direction depends on LTR normalization (home defends x=0, away defends x=105).

## 5. Blocking Score Primitive

### 5.1 Threat Model (Grid-Based Voronoi Sum)

For a given frame with attacking team in possession, threat is a grid-based sum
following the paper's approach (not point evaluation at receiver positions):

1. **Compute surfaces once.** `compute_pitch_control(frame)` returns a
   `PitchControlSurface` with grid shape `(ny, nx)` (default 32×50).
   `ExpectedThreat.interpolator()` produces xT values at the same grid coordinates.
   Element-wise: `threat_grid[i,j] = xT(x_j, y_i) × PC(y_i, x_j)`.

2. **Voronoi partition over ALL attackers.** For each grid cell, assign it to the
   nearest attacking player (Euclidean distance) — including non-dangerous (behind-ball)
   receivers. This matches the paper's RQ2 which partitions using all attackers.

3. **Sum only over dangerous receivers' regions.**
   ```
   threat_r = sum_{(i,j) in N_r}  threat_grid[i,j]
   threat(frame) = sum_{r in dangerous}  threat_r
   ```
   Non-dangerous receivers' Voronoi regions are computed but their threat sums are
   ignored. This preserves correct spatial resolution around dangerous receivers
   (their Voronoi regions are not inflated by absorbing behind-ball cells) while still
   excluding low-value behind-ball threat from the total.

   Note: `cell_area` is omitted from the sum. On a uniform grid it is a constant that
   cancels in both the `blocking_score` delta and `blocked_threat_fraction` ratio.
   Omitting it keeps blocking_score magnitudes comparable to the paper's reported values.

This matches the paper's formula: `threat_r(S) = Σ_{(x,y) ∈ N_r} threat(x,y) · p_r(x,y)`.
The paper notes: "Passes often target the open space near a player rather than their
exact location. Therefore, both aspects should not only consider the receiver's exact
location, but also his surroundings."

The Voronoi assignment is O(n_cells × n_attackers) ≈ O(1600 × 10) = ~16000 ops —
negligible relative to the pitch control computation. No extra PC calls are needed.

This substitutes the paper's `xT × SoccerMap(pass_selection) × SoccerMap(pass_success)`
with `xT × PC_share`. Pitch control captures spatial accessibility (roughly analogous
to pass success probability). SoccerMap requires a CNN architecture, 1M+ training
passes, and PyTorch/TF runtime — infeasible for a pure pandas/numpy/sklearn library.

### 5.2 Counterfactual Construction

Remove defenders from the frame DataFrame and recompute:

```
threat_unblocked = threat(frame_without_removed_defenders)
threat_original  = threat(frame)
blocking_score   = threat_unblocked - threat_original
```

Positive blocking_score = defenders are suppressing threat.

### 5.3 Function Signature

```python
def compute_blocking_score(
    frame: pd.DataFrame,
    attacking_team_id: int | str,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    defenders_to_remove: list[int | str] | None = None,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    params: PitchControlParams | None = None,
) -> float:
```

**LTR validation:** `compute_blocking_score` raises `ValueError` on non-LTR frames
(same guard as `lane_control` — see §4.7).

| `defenders_to_remove` | Behavior                                              | PC calls |
|------------------------|-------------------------------------------------------|----------|
| `None`                 | Auto-identify lane-blockers via man-marking filter, remove all | 2 |
| `[player_id]`          | Remove exactly that player                            | 2        |
| `[pid_1, pid_2, ...]`  | Remove exactly those players                          | 2        |

Always 2 pitch control calls regardless. The caller controls granularity:

- **VAEP aggregator** calls once with `defenders_to_remove=None` -> team-level score
- **TF-19 GKDV** calls with `defenders_to_remove=[gk_id]` -> GK-specific score
- **Coaching analysis** loops over individual defenders -> per-defender breakdown

### 5.4 Dangerous Receiver Selection

Only receivers positioned between the ball and the defending goal:

```python
if attacking_toward_high_x:
    dangerous = attackers[attackers["x"] > ball_x]
else:
    dangerous = attackers[attackers["x"] < ball_x]
```

GK excluded from receiver set.

### 5.5 Caching Note

When `defenders_to_remove=None`, the function internally identifies lane-blockers by
running `lane_control` for each dangerous receiver. The action-coupled aggregator will
pre-compute lane control results and pass identified lane-blockers explicitly rather
than recomputing.

### 5.6 Known Ball Physics Inconsistency

`lane_control` uses the Spearman quadratic drag model (`T_ball = expm1(k×d)/(v₀×k)`,
§4.2) for pass-specific interception timing. `compute_blocking_score` internally calls
`compute_pitch_control`, which uses a constant-speed ball model at 15 m/s
(`_spearman.py:180-188` — `ball_dist / params.average_ball_speed`, no drag).

These models answer different physical questions:

- **Lane control:** "Can a defender intercept THIS specific pass trajectory?" → drag
  model gives realistic deceleration along the pass path.
- **Pitch control:** "Who controls this region of the pitch in general?" → constant
  speed is a simplification for spatial dominance, not trajectory-specific timing.

The counterfactual delta (`threat_unblocked - threat_original`) is **self-consistent**:
both the original and counterfactual pitch control surfaces use the same constant-speed
model, so the ball physics cancel in the difference. The inconsistency means the
absolute threat values are approximate, but the blocking_score delta is stable.

If a future silly-kicks version upgrades pitch control to a drag-aware ball model, the
blocking score computation benefits automatically with no API change.

## 6. Action-Coupled Layer

### 6.1 Aggregator

```python
@nan_safe_enrichment
def add_cover_shadows(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    decision_rule: Literal["any", "majority", "all"] = "majority",
    detailed: bool = False,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
) -> pd.DataFrame:
```

Per-action logic:

1. Link action to frame via `link_actions_to_frames`
2. Identify passer (action's `player_id`) and position (`start_x`, `start_y`)
3. Identify dangerous receivers (forward attacking teammates in the linked frame)
4. Identify lane-blocking defenders (non-man-marking opponents via filter)
5. Run `lane_control` for each (passer, receiver) pair -> `LaneControlResult`
6. Run `compute_blocking_score` with identified lane-blockers -> team-level score
7. Compute `max_single_defender_blocking_score`:
   - `detailed=False` (default): lightweight approximation from Lane Control —
     `max over defenders d of: sum_r xT(r) x delta_P_received_r` where delta_P is
     the change in lane_control receive probability when defender d is excluded
     from the TTI race. This is a novel approximation not from the paper — validated
     via rank-correlation test against the full counterfactual mode (§9.2,
     `test_detailed_vs_lightweight_rank_correlation`, target: Spearman ρ ≥ 0.7).
   - `detailed=True`: loop over each lane-blocker calling `compute_blocking_score`
     with `defenders_to_remove=[d]` -> true per-defender PC counterfactual (matches
     the paper's approach)

### 6.2 Output Columns (5)

| Column                              | Type    | Source                            | NaN semantics                      |
|-------------------------------------|---------|-----------------------------------|------------------------------------|
| `n_blocked_receivers`               | Int64   | Count of receivers whose lanes are blocked per `decision_rule` | pd.NA if unlinked / no receivers   |
| `n_potential_receivers`             | Int64   | Count of dangerous receivers      | pd.NA if unlinked                  |
| `blocking_score`                    | float64 | Team-level counterfactual         | NaN if unlinked                    |
| `blocked_threat_fraction`           | float64 | blocking_score / threat_unblocked | NaN if unlinked                    |
| `max_single_defender_blocking_score`| float64 | Per-defender max (lightweight/full)| NaN if unlinked / no lane-blockers |

All columns NaN/pd.NA for actions that cannot link to a frame. All 0.0 for actions
that link but have no dangerous receivers or no lane-blocking defenders.

**Edge case:** `blocked_threat_fraction` returns 0.0 (not NaN) when `threat_unblocked ≤ 0`
(no threat means no fraction blocked — division by zero guard).

### 6.3 VAEP Factory

```python
def cover_shadow_xfns(
    xt: ExpectedThreat,
    *,
    home_team_id: int | str,
    decision_rule: Literal["any", "majority", "all"] = "majority",
    detailed: bool = False,
    method: str = "spearman",
) -> list:
```

Returns a list with ONE `FrameAwareTransformer` emitting 5 columns x 3 game states =
15 VAEP columns. Frame-precomputation cache keyed on `(period_id, frame_id, team_id)`
to avoid redundant computation across game-state slots.

The transformer handles the VAEP introspection contract (10-row dummy gamestate with
17 canonical SPADL columns -> silent NaN, per PR-S21 pattern).

### 6.4 Atomic Mirror

`silly_kicks/atomic/tracking/features.py` gets the same surface — `add_cover_shadows`
+ `cover_shadow_xfns` — anchored on atomic SPADL's `x/y` instead of standard's
`start_x/start_y`. Mechanical mirroring per existing pattern.

### 6.5 Decision Rule Default

`decision_rule="majority"` (default). Paper's Table 1 shows Majority has the best
balanced accuracy (68.01% vs 66.93% for All, vs lower for Any). The parameter lives
on the aggregator/factory signature, not in `CoverShadowParams` — it is a consumer-side
interpretation choice, not a physics parameter.

## 7. Parameters

### 7.1 CoverShadowParams

```python
@dataclass(frozen=True)
class CoverShadowParams:
    # Corridor parameterization
    n_sample_points: int = 30
    cone_width_factor: float = 0.2

    # Ball drag model (Spearman 2017)
    air_density: float = 1.22           # kg/m^3
    drag_coefficient: float = 0.25      # C_D
    ball_cross_section: float = 0.038   # m^2
    ball_mass: float = 0.42             # kg
    ball_initial_speed: float = 12.0    # m/s

    # Player TTI
    reaction_time: float = 0.7          # seconds
    max_acceleration: float = 7.0       # m/s^2
    max_speed: float = 12.0             # m/s
    block_radius: float = 0.7           # meters (defenders only)

    # Probability conversion
    sigma: float = 0.20                 # sigmoid width
    lambda_ctrl: float = 4.3            # control rate (s^-1)

    # Man-marking filter
    man_mark_radius: float = 3.0        # meters
    man_mark_behind_offset: float = 1.0 # meters behind attacker
```

Derived constant `k_drag` as a `@property`:

```python
@property
def k_drag(self) -> float:
    return (self.air_density * self.drag_coefficient
            * self.ball_cross_section) / (2 * self.ball_mass)
```

### 7.2 No Post-Init Validation

All fields are independent — no invalid combinations. Range checks omitted, matching
existing param dataclasses (`SpearmanParams`, `AndrienkoParams`).

### 7.3 Parameter Tuning (TF-24 relevance)

The paper calibrated σ=0.20 and λ=4.3 on StatsBomb 360 freeze frames with estimated
velocities (position deltas between consecutive events — noisy, event-frequency
sampling). silly-kicks uses full tracking data with measured velocities at 25 Hz — a
fundamentally different velocity noise profile. Parameter transfer needs empirical
validation.

**Calibration is explicitly deferred to TF-24** (Optuna calibration infrastructure).
`CoverShadowParams` is a clean Optuna target. In the interim, TF-30 ships with the
paper's defaults and includes a predicted-block-rate smoke test (§9.2) — the fraction
of (passer, receiver) pairs classified as blocked under Majority rule on provider
fixtures. This is NOT false-positive rate (which requires ground-truth pass outcomes we
don't have in fixtures). The test asserts the predicted block rate falls in a plausible
range (e.g., 10-60%) as a sanity check, not a calibration assertion.

## 8. Threat Model Substitution Rationale

The paper uses `xT x SoccerMap(pass_selection) x SoccerMap(pass_success)`. Each
SoccerMap is a CNN (Fernandez & Bornn 2021) trained on 1.1M passes with freeze frames,
requiring PyTorch/TF, ~50-100MB model weights, and league/provider-specific training.
This is infeasible for silly-kicks (pure pandas/numpy/sklearn, no neural network deps).

Our substitution `xT x pitch_control`:

- **xT** — same underlying framework as the paper (Karun Singh's Expected Threat).
  The paper uses Singh's pre-computed 12×8 grid (2017-18 EPL) bilinear-upscaled to
  104×68. silly-kicks `ExpectedThreat` is fitted from actual data at configurable
  resolution (default 16×12, `xthreat.py:M=12, N=16`). The cover shadow computation
  uses `ExpectedThreat.interpolator()` to evaluate xT at pitch_control grid
  coordinates, so the native xT resolution is irrelevant — values are interpolated
  to the PC grid.
- **pitch_control** — captures spatial accessibility in each receiver's Voronoi region
  (§5.1). Removing a defender shifts the pitch control surface, increasing the
  attacking team's share at newly-opened positions. This is roughly analogous to the
  pass-success probability changing when a blocking defender is removed.
- **pass selection** is implicitly captured via the Voronoi partitioning: each
  receiver's threat is the spatial integral over their surrounding region weighted by
  xT, so high-threat regions dominate the allocation.

The substitution is acknowledged as a simplification. The paper's SoccerMap approach
is strictly more expressive (context-dependent per-cell probabilities). If a future
silly-kicks version adds a neural pass model, `compute_blocking_score` can accept a
`threat_fn` callback without API change.

## 9. Testing Strategy

### 9.1 Three-Tier Fixture Strategy

| Tier                    | Fixtures                                         | What it tests                                                  | Velocity source                |
|-------------------------|--------------------------------------------------|----------------------------------------------------------------|--------------------------------|
| Synthetic               | Constructed in-test with explicit pos + vel       | Physics correctness — TTI branches, probability, man-marking   | Hand-set vx/vy                 |
| Provider-parameterized  | `{provider}_slim.parquet` x 4, + smooth/derive   | Cross-provider shape/dtype/NaN-rate; player_id dtype asymmetry | Derived from position deltas   |
| E2e                     | Full lakehouse datasets, `@pytest.mark.e2e`      | Real-world blocking rates, threat ranges, perf budget          | Native or derived              |

### 9.2 Synthetic Tests (`tests/tracking/test_cover_shadows.py`)

| Test                                    | What it verifies                                                       |
|-----------------------------------------|------------------------------------------------------------------------|
| `test_ball_drag_time_known_distances`   | Ball drag model against hand-computed values (d=0 -> t=0, d=10 -> verify) |
| `test_player_tti_three_phases`          | All 3 piecewise branches: cruising, acceleration-only, accel+cruise    |
| `test_player_tti_block_radius`          | Defender 0.7m closer effectively than attacker at same position        |
| `test_lane_control_defender_on_line`    | Defender on center line -> all 3 lines blocked                         |
| `test_lane_control_defender_off_line`   | Defender far off to side -> all 3 lines open                           |
| `test_lane_control_fast_defender`       | Moving defender intercepts despite not being on line                   |
| `test_lane_control_decision_rules`      | Exactly 1 line blocked -> any=True, majority=False, all=False          |
| `test_man_marking_filter`              | 2m behind attacker -> man-marker; 5m laterally -> lane-blocker         |
| `test_blocking_score_no_lane_blockers`  | All defenders man-marking -> blocking_score = 0.0                      |
| `test_blocking_score_positive`          | Lane-blocker removed -> threat increases -> positive score             |
| `test_blocking_score_specific_defender` | `defenders_to_remove=[pid]` removes exactly that player                |
| `test_blocking_score_no_receivers`      | All attackers behind ball -> blocking_score = 0.0                      |
| `test_add_cover_shadows_columns`        | Returns all 5 columns with correct dtypes                              |
| `test_add_cover_shadows_unlinked_nan`   | Unlinked actions -> NaN/pd.NA in all columns                           |
| `test_add_cover_shadows_detailed_flag`  | Both modes run; detailed=True >= detailed=False for max per-defender    |
| `test_detailed_vs_lightweight_rank_correlation` | Spearman rank correlation ≥ 0.7 between detailed=True and False per-defender scores on a multi-defender scenario |
| `test_cover_shadow_xfns_introspection`  | 10-row dummy -> silent NaN (VAEP fit-time contract)                    |
| `test_cover_shadow_xfns_column_count`   | 5 features x 3 states = 15 output columns                             |
| `test_cover_shadow_tti_subsumes_compute_tti` | New 3-phase TTI with max_speed=1e6, block_radius=0 matches compute_tti within 1e-6 on identical inputs |
| `test_blocking_rate_smoke`              | Predicted block rate (fraction of pairs classified blocked under Majority) in plausible range 10-60% — sanity, not calibration (see §7.3) |
| `test_params_drift_guard`              | CoverShadowParams().reaction_time == SpearmanParams().reaction_time AND max_acceleration match |
| `test_voronoi_partition_covers_grid`    | Voronoi over all attackers assigns every grid cell to exactly one player; dangerous receivers' regions are smaller than when non-dangerous are excluded |
| `test_threat_grid_matches_point_eval_single_receiver` | With 1 receiver (Voronoi = entire grid), grid sum ≥ point evaluation — validates spatial capture |

### 9.3 Provider-Parameterized Tests (`tests/tracking/test_cover_shadows_providers.py`)

All 4 providers: Sportec, Metrica, SkillCorner, PFF.

Frames loaded via `load_provider_frames()` then preprocessed with
`smooth_frames() + derive_velocities()` (established pattern from Bekkers/DAS/GK
influence tests — existing slim fixtures lack vx/vy).

| Test                                         | What it verifies                                                    |
|----------------------------------------------|---------------------------------------------------------------------|
| `test_add_cover_shadows_shape_and_dtypes`    | 5 columns present, correct dtypes, no crashes per provider          |
| `test_cover_shadows_nan_rate_bounds`         | NaN rate < provider-specific ceiling (relaxed for Metrica ball gaps) |
| `test_cover_shadows_value_bounds`            | blocking_score >= 0, blocked_threat_fraction in [0,1], etc.         |
| `test_n_valid_blocked_receivers_nonzero`      | At least 1 action has n_blocked_receivers >= 1 (anti-vacuous gate)      |

Provider-specific edge cases:

| Provider    | Edge case                                             |
|-------------|-------------------------------------------------------|
| Sportec     | int64 player_ids, 25 Hz native frame rate             |
| Metrica     | object/string player_ids, ~77% NaN ball coords        |
| SkillCorner | object/string player_ids, variable frame rate         |
| PFF         | Int64 (nullable pandas) player_ids, separate fixture  |

### 9.4 Fixture Regeneration

If existing slim fixtures lack sufficient geometric diversity for cover shadow testing
(no frames where a defender sits between passer and receiver in the cone), regenerate
from raw ingestion data. Pattern: extend `synthesize_actions()` in
`_provider_inputs.py` with a `CoverShadowScenario` dataclass selecting frames where at
least one defender lies geometrically between passer and a forward teammate within the
cone. The scenario selection queries real frame geometry — if no provider's slim fixture
has a naturally occurring cover shadow geometry, regenerate the slim fixture from a
different match slice that does.

### 9.5 Invariant Tests (`tests/invariants/test_cover_shadow_invariants.py`)

| Invariant                              | Property                                              |
|----------------------------------------|-------------------------------------------------------|
| `blocking_score >= 0`                  | Removing defenders cannot decrease threat (monotonicity) |
| `blocked_threat_fraction in [0, 1]`    | Bounded by definition                                 |
| `n_blocked_receivers <= n_potential_receivers` | Cannot block more lanes than exist                  |
| `n_blocked_receivers >= 0`                 | Integer, non-negative                                 |
| `blocking_score ~ 0 when n_blocked_receivers = 0` | No blocked lanes -> minimal threat reduction (approx) |

### 9.6 Performance Budget

Budgets assume `method="spearman"` (default). Voronoi is cheaper; Fernandez-Bornn is
comparable. All three methods must fit within the stated budget.

| Scope                       | Budget        | Bottleneck                                             |
|-----------------------------|---------------|--------------------------------------------------------|
| Per action (`detailed=False`) | ≤ 500 ms    | 2 × `compute_pitch_control` (~50-200ms each) + lane control per receiver |
| Per match (`detailed=False`)  | ≤ 5 min     | ~500 pass actions × 2 PC calls each                   |
| Per action (`detailed=True`)  | ≤ 1.5 s     | (2 + N_lane_blockers) × PC calls; ~3 blockers typical |
| Per match (`detailed=True`)   | N/A         | Intended for single-frame coaching analysis, not full-match VAEP |

The VAEP factory caches PC surfaces keyed on `(period_id, frame_id)` — when multiple
game-state slots reference the same frame, the original PC surface is reused. The xT
grid is pre-interpolated once to the PC grid shape at factory initialization.

Lane control per (passer, receiver) pair is cheap: 3 lines × 30 points × n_players ≈
3 × 30 × 22 = ~2000 TTI evaluations (vectorized numpy, <1ms).

E2e tests assert per-match wall time against the `detailed=False` budgets with 1.5×
headroom for CI runner variance (per feedback: use flat ceilings, not platform ternaries).

### 9.7 Additional Test Files

- `tests/tracking/test_cover_shadows_e2e.py` — `@pytest.mark.e2e`, full pipeline
  with real lakehouse data
- `tests/atomic/tracking/test_cover_shadows_atomic.py` — atomic mirror parity

## 10. NOTICE Entry

```
The cover shadow features in silly_kicks/tracking/_cover_shadows.py (PR-S36,
TF-30) implement methodologies described in:

- Cascioli, L., Wang, A., Stradiotti, L., Van Roy, M., Robberechts, P.,
  Wouters, M., Jaspers, A., & Davis, J. (2025). "Quantifying Off-Ball
  Defensive Impact through Cover Shadows." Hudl Research / DTAI, KU Leuven.
  (Lane Control physics-based pass-blocking model; blocking score
  counterfactual threat reduction metric)

- Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017).
  "Physics-Based Modeling of Pass Probabilities in Soccer." MIT Sloan SAC.
  (Ball drag model: quadratic air resistance with rho=1.22, C_D=0.25,
  A=0.038, m=0.42; referenced by Cascioli et al. for ball travel time)
```

ADR-005 amendment adds cover shadows to the `_frame_aware` feature registry.

## 11. Complements Existing Features

| Feature                | Mechanism                      | Complementarity                                         |
|------------------------|--------------------------------|---------------------------------------------------------|
| Pressure (TF-2)        | Proximity-based on ball carrier | Cover shadows measure lane-blocking on passing OPTIONS  |
| Off-ball runs (TF-4)   | Attacking movement detection   | Cover shadows identify defensive response nullifying it |
| Defensive line (TF-14) | Back-line geometry             | Provides "attackers between ball and goal" filter       |
| GK influence (TF-15)   | GK spatial contribution        | GK-specific blocking_score feeds TF-19 GKDV            |
| DAS (TF-28)            | Physics-based space valuation  | Independent cross-check on pitch-control counterfactual |

## 12. GKDV Integration (TF-19)

`compute_blocking_score(frame, ..., defenders_to_remove=[gk_id])` provides the
per-GK lane-specific deterrent signal for GKDV Layer 3:

```
gk_lane_deterrent = blocking_score(actual_GK) - blocking_score(ghost_GK)
```

The primitive supports this from day 1 via `defenders_to_remove`. No action-coupled
layer changes needed — TF-19 calls the primitive directly.

## 13. LOC Estimates

| Location                        | LOC   |
|---------------------------------|-------|
| `_cover_shadows.py`             | ~400  |
| `features.py` additions        | ~120  |
| `atomic/tracking/features.py`  | ~30   |
| Tests (all files)               | ~500  |
| NOTICE + ADR-005 amendment      | ~20   |
| **Total**                       | ~1070 |
