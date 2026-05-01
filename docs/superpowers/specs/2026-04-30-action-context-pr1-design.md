# Tracking-aware features — PR-1 (`action_context`) — silly-kicks 2.8.0

**Status:** Approved (design)
**Target release:** silly-kicks 2.8.0
**Author:** Karsten S. Nielsen with Claude Opus 4.7 (1M)
**Date:** 2026-04-30
**Predecessor:** silly-kicks 2.7.0 — `silly_kicks.tracking` namespace primitive layer (PR-S19; ADR-004)
**Successor:** PR-S21 — `pre_shot_gk_position_*` refining `add_pre_shot_gk_context` (Dunkin'; GK-coach priority)

**Triggers:**

- ADR-004 invariant 9, deferred-priority item #1: `action_context()` tracking-aware VAEP/xG features.
- Lakehouse cross-check session (2026-04-30) recommendation: ship `nearest_defender_distance` + `actor_speed` + `receiver_zone_density` as the impact-bundle; pressure deferred per its own scoping cycle.
- Empirical-validation probe (2026-04-30): the lakehouse does NOT precompute these features today. PR-S20 establishes the baseline implementation; future lakehouse boundary adapters will consume from silly-kicks.
- Best-practice, long-term TDD/E2E discipline: full atomic-SPADL parity from day one; no cross-namespace asymmetry.
- Academic-attribution discipline: every shipped feature carries explicit references to its original literature, mirroring how the lakehouse cites Pollard, Spearman, Lucey, Anzer & Bauer, Power et al., Decroos et al. in code, model cards, TODO entries, and ADRs. Each public function's docstring includes a "References" section; the TODO.md On-Deck table cites the originating paper for each deferred item.

---

## 1. Problem

silly-kicks 2.7.0 (PR-S19) shipped the `silly_kicks.tracking` namespace primitive layer: 19-column long-form schema, 4 provider adapters (PFF, Sportec, Metrica, SkillCorner), `link_actions_to_frames` + `slice_around_event` linkage primitives, and `play_left_to_right` for tracking. Per ADR-004, tracking-aware *features* were explicitly deferred to follow-up scoping cycles — PR-S19 ships nothing that consumes the linkage primitive.

Without tracking-aware features, the tracking namespace ships unused. The library shape is correct (per the 9-invariant ADR-004 charter) but no model, no analytic, no downstream pipeline exercises the linkage primitive on real data. Tracking-aware features are the load-bearing reason the namespace exists; PR-1 (this spec) is the first PR that delivers value through it.

The lakehouse 2026-04-30 cross-check session ranked `action_context()` as priority #1 by `(lakehouse leverage × analytical novelty) ÷ POC scope`, with three sub-features all reducing to "frame-lookup + measure": nearest-defender distance, actor speed, receiver-zone density. The lakehouse-internal probe (2026-04-30) confirms these features are NOT precomputed in any existing lakehouse mart — the lakehouse will consume from silly-kicks once PR-S20 ships, not the other way.

This PR (PR-S20) lands the **first tracking-aware feature set**: 4 features built on the locked PR-S19 schema + linkage primitive, with full VAEP/HybridVAEP/AtomicVAEP integration via a long-term-best architecture that extends the existing `xfns` mechanism rather than introducing a class-per-feature-kind hierarchy. A new ADR-005 codifies the integration contract so PR-S21+ tracking-aware features inherit it without re-litigation.

## 2. Goals

1. **4-feature catalog**, all sharing one compute kernel pattern (link → cdist over opposite-team rows from linked frame → geometric aggregate):
   - `nearest_defender_distance` — meters to closest opposing-team player at action start
   - `actor_speed` — m/s of the action's `player_id` at the linked frame
   - `receiver_zone_density` — count of opposing-team players within radius R of action end
   - `defenders_in_triangle_to_goal` — count of opposing-team players inside the wedge formed by action start → goal posts
2. **Two public surfaces** for the catalog: per-feature primitives `(actions, frames) → pd.Series` and aggregator `add_action_context(actions, frames, **opts) → pd.DataFrame` enriching actions with the 4 features + 4 linkage-provenance columns.
3. **Full atomic-SPADL parity** at PR-S20 ship time. Standard SPADL and Atomic SPADL each get their own `tracking.features` module sharing a single private kernel module (`silly_kicks/tracking/_kernels.py`).
4. **VAEP / HybridVAEP / AtomicVAEP integration** via composition: extend `VAEP.compute_features` to accept an optional `frames=None` kwarg + dispatch on a `_frame_aware` xfn marker. No new VAEP subclass; HybridVAEP and AtomicVAEP inherit the extension. Users opt in by appending `tracking_default_xfns` to their xfns list.
5. **`lift_to_states` extension utility** shipped alongside the per-feature helpers. Lifts an `(actions, frames) → pd.Series` helper to a `(states, frames) → Features` transformer with `_a0/_a1/_a2` columns. Proves the extensibility contract from Q2 with a passing test rather than asserting it in prose.
6. **TDD-first**, RED-before-GREEN, ~13 implementation loops (preceded by Loop 0 lakehouse probe). Empirical-validation deliverable in Loop 9 + Loop 11: tracking-augmented model AUC ≥ baseline AUC + ε on synthetic fixtures.
7. **Three-tier CI fixture strategy**: in-memory analytical (Tier 1, the bulk), retained PR-S19 parquets (Tier 2, cross-provider linkage), new lakehouse-derived slim slices (Tier 3, real-data cross-validation for Sportec/Metrica/SkillCorner; PFF synthetic-only per license).
8. **ADR-005 charter** capturing the 7 cross-cutting decisions PR-S20 introduces, so PR-S21+ tracking-aware features inherit them.
9. **TODO.md restructure** to mirror lakehouse-style "On Deck" table format. Each deferred follow-up becomes an explicit On-Deck item with Size/#/Task/Source/Notes columns including academic citations.
10. **Academic attribution baked in.** Every public feature docstring includes a `References` section. The TODO.md On-Deck Notes column cites the originating literature. The `add_action_context` aggregator and `tracking_default_xfns` list reference the impact-bundle literature in their docstrings. Mirrors the lakehouse's discipline of citing Pollard, Spearman, Lucey, Anzer & Bauer, Power, Decroos, Bekkers, etc. wherever those works inform the implementation.
11. **`NOTICE` file established at repo root**, mirroring the lakehouse's `NOTICE` pattern (Third-Party Libraries + Mathematical References sections). silly-kicks does not currently have a NOTICE file; PR-S20 is the right time to establish the canonical pattern because it's the first PR introducing distinct academic methodologies beyond the foundational VAEP/SPADL/xT body. The PR-S20 NOTICE seeds the file with all currently-known academic dependencies (not just PR-S20's 4 features) so future PRs extend rather than retrofit. Cross-linked from `README.md` and `CLAUDE.md`.

## 3. Non-goals

- **No new tracking-aware features beyond the 4 locked.** Pressure (`pressure_on_actor`), pre-shot GK refinement, off-ball runs, actor pre-window distance, ball-carrier inference, sync_score, pitch-control, smoothing, gap-filling — all explicitly deferred per the On-Deck table (§8). TF-1 is `pre_shot_gk_position_*` per GK-coach priority.
- **No StatsBomb / Opta / Wyscout / kloppy-events provider tracking work.** PR-S20 consumes already-converted SPADL actions and tracking frames. The 6 events providers and the kloppy gateway remain unchanged.
- **No new tracking provider adapters.** PR-S20 consumes the 4-provider coverage from PR-S19; ReSpo.Vision remains licensing-blocked.
- **No lakehouse boundary adapter.** PR-S20 ships the library primitives; lakehouse pipelines build their own boundary in their own repo. The cross-validation in L3b uses lakehouse-derived slim parquets, not a live lakehouse SQL boundary.
- **No formula choice for pressure.** Pressure ships as PR-S22 (TF-2) with its own scoping cycle (Andrienko vs. Link vs. Bekkers DEFCON). PR-S20 does not ship pressure.
- **No streaming / chunked processing.** Per ADR-004 invariant 1 (hexagonal pure-function contract). Out-of-core handling is downstream's concern.
- **No env-var-gated tests in CI.** L3b real-data sweep (`tests/test_action_context_real_data_sweep.py`) is `e2e`-marked and skips with explicit reason on missing local data.
- **No new wide-form / 120×80 / object-id schema variants.** PR-S20 inherits PR-S19's locked schema verbatim.

## 4. Design

### 4.1 Architecture & module layout

```
silly_kicks/
├── tracking/
│   ├── __init__.py              # extend exports for new modules
│   ├── _kernels.py              # NEW — pure compute kernels (private; schema-agnostic)
│   ├── feature_framework.py     # NEW — ActionFrameContext, lift_to_states, type aliases
│   ├── features.py              # NEW — standard SPADL public surface (4 fns + aggregator + default_xfns)
│   ├── utils.py                 # extend: _resolve_action_frame_context (private helper)
│   └── (existing schema.py / sportec.py / pff.py / kloppy.py / _direction.py unchanged)
├── atomic/
│   ├── tracking/                # NEW package
│   │   ├── __init__.py
│   │   └── features.py          # atomic SPADL public surface — wraps shared _kernels
│   └── (existing atomic structure unchanged)
├── vaep/
│   ├── base.py                  # extend: VAEP.compute_features(*, frames=None), VAEP.rate(*, frames=None)
│   ├── feature_framework.py     # extend: frame_aware decorator, is_frame_aware helper, Frames type alias, FrameAwareTransformer
│   └── (HybridVAEP, AtomicVAEP — zero changes; inherit base extension)
```

**Direction-of-dependency:** `tracking → vaep` for the marker convention (`silly_kicks.tracking.feature_framework` imports `frame_aware` from `silly_kicks.vaep.feature_framework`). `vaep → tracking` only via lazy import inside the `if frames is not None` branch in `VAEP.compute_features` (for the symmetric `tracking.play_left_to_right` call). No module-import-time coupling.

**Schema-aware-via-per-namespace-wrapper.** Standard SPADL has columns `start_x, start_y, end_x, end_y`; atomic SPADL has `x, y, dx, dy`. A single private `_kernels.py` exposes schema-agnostic compute functions taking `(anchor_x, anchor_y, ctx)` directly; per-schema wrapper modules read the right columns and call the kernel. This pattern parallels how `silly_kicks.atomic.spadl.utils` mirrors `silly_kicks.spadl.utils` (existing repo precedent) while extracting kernel logic to a single source of truth (long-term DRY for PR-S21+ tracking features).

### 4.2 Public API

#### 4.2.1 Per-feature helpers (standard SPADL, in `silly_kicks/tracking/features.py`)

Each public function's full docstring includes a `References` section citing the canonical literature. The signatures below show only the type contract; complete docstrings are written during Loop 7.

```python
def nearest_defender_distance(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """meters to closest opposing-team player at the linked frame.
    Anchor: (action.start_x, action.start_y). NaN if action couldn't link.

    References
    ----------
    Lucey, P., Bialkowski, A., Monfort, M., Carr, P., & Matthews, I. (2014).
        "Quality vs Quantity: Improved Shot Prediction in Soccer using Strategic
        Features from Spatiotemporal Data." MIT Sloan Sports Analytics Conference.
    Anzer, G., & Bauer, P. (2021). "A goal scoring probability model for shots
        based on synchronized positional and event data in football and futsal."
        Frontiers in Sports and Active Living, 3, 624475.
    """

def actor_speed(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """m/s of the action's player_id at the linked frame.
    NaN if action couldn't link, actor's player_id missing from frame, or speed is NaN.

    References
    ----------
    Anzer, G., & Bauer, P. (2021). "A goal scoring probability model for shots
        based on synchronized positional and event data in football and futsal."
        Frontiers in Sports and Active Living, 3, 624475. (player_speed as xG feature)
    Bauer, P., & Anzer, G. (2021). "Data-driven detection of counterpressing in
        professional football." Data Mining and Knowledge Discovery, 35(5), 2009-2049.
    """

def receiver_zone_density(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    radius: float = 5.0,
) -> pd.Series:
    """count of opposing-team players within radius of (action.end_x, action.end_y).
    Integer-valued (0 if linked but no defenders in radius; NaN if unlinked).

    References
    ----------
    Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan Sports Analytics Conference.
        (zone-based defender intensity in pitch-control framework)
    Power, P., Ruiz, H., Wei, X., & Lucey, P. (2017). "Not all passes are created
        equal: Objectively measuring the risk and reward of passes in soccer from
        tracking data." KDD '17 (OBSO).
    """

def defenders_in_triangle_to_goal(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series:
    """count of opposing-team players inside the triangle (action.start_x, action.start_y) →
    goal-mouth posts at x=105 (left post y=30.34, right post y=37.66 per spadl.config).
    Integer-valued; NaN if unlinked.

    References
    ----------
    Lucey, P., Bialkowski, A., Monfort, M., Carr, P., & Matthews, I. (2014).
        "Quality vs Quantity: Improved Shot Prediction in Soccer using Strategic
        Features from Spatiotemporal Data." MIT Sloan Sports Analytics Conference.
        (canonical "defenders in shot triangle" feature)
    Pollard, R., & Reep, C. (1997). "Measuring the effectiveness of playing strategies
        at soccer." Journal of the Royal Statistical Society Series D, 46(4), 541-550.
        (early shot-quality / pressure-from-defenders concept)
    """
```

#### 4.2.2 Aggregator (`add_action_context`)

```python
@nan_safe_enrichment
def add_action_context(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    receiver_zone_radius: float = 5.0,
) -> pd.DataFrame:
    """Enrich actions with 4 tracking-aware features + 4 linkage-provenance columns.

    Returns
    -------
    pd.DataFrame
        Input actions plus the columns:
        - nearest_defender_distance (float64, meters)
        - actor_speed (float64, m/s)
        - receiver_zone_density (Int64, count)
        - defenders_in_triangle_to_goal (Int64, count)
        - frame_id (Int64; NaN if unlinked)
        - time_offset_seconds (float64; NaN if unlinked)
        - link_quality_score (float64; NaN if unlinked)
        - n_candidate_frames (int64)
    """
```

#### 4.2.3 Atomic SPADL surface (`silly_kicks/atomic/tracking/features.py`)

Same public function names + signatures as standard. Internally reads `(x, y)` for start-anchor and `(x + dx, y + dy)` for end-anchor. Calls the shared `_kernels` functions.

#### 4.2.4 VAEP integration

```python
# silly_kicks/vaep/feature_framework.py — additive
def frame_aware(fn: Callable) -> Callable:
    """Marker decorator: this xfn requires frames as the second argument."""
    fn._frame_aware = True
    return fn

def is_frame_aware(fn: Callable) -> bool:
    return getattr(fn, "_frame_aware", False)

Frames = pd.DataFrame
FrameAwareTransformer = Callable[[GameStates, Frames], Features]
```

```python
# silly_kicks/vaep/base.py — additive kwarg
class VAEP:
    def compute_features(
        self,
        game: pd.Series,
        game_actions: fs.Actions,
        *,
        frames: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        game_actions_with_names = self._add_names(game_actions)
        states = self._fs.gamestates(game_actions_with_names, self.nb_prev_actions)
        states = self._fs.play_left_to_right(states, game.home_team_id)

        if frames is not None:
            from silly_kicks.tracking.utils import play_left_to_right as _track_ltr
            frames = _track_ltr(frames, game.home_team_id)

        feats = []
        for fn in self.xfns:
            if is_frame_aware(fn):
                if frames is None:
                    raise ValueError(
                        f"{fn.__name__} requires frames; pass frames= to compute_features"
                    )
                feats.append(fn(states, frames))
            else:
                feats.append(fn(states))
        return pd.concat(feats, axis=1)

    def rate(
        self,
        game: pd.Series,
        game_actions: fs.Actions,
        game_states: fs.Features | None = None,
        *,
        frames: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # ... existing logic; if game_states is None, pass frames= through to compute_features
```

`HybridVAEP` and `AtomicVAEP` inherit the extension automatically — zero code changes in `silly_kicks/vaep/hybrid.py` or `silly_kicks/atomic/vaep/base.py`.

#### 4.2.5 `lift_to_states` and default xfn lists

```python
# silly_kicks/tracking/feature_framework.py
def lift_to_states(
    helper: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    nb_states: int = 3,
) -> FrameAwareTransformer:
    """Lift a (actions, frames) -> Series helper to a (states, frames) -> Features transformer.
    Output columns: f"{helper.__name__}_a0", ..._a{nb_states-1}.
    Marks the returned transformer as _frame_aware=True."""

# silly_kicks/tracking/features.py
tracking_default_xfns = [
    lift_to_states(nearest_defender_distance),
    lift_to_states(actor_speed),
    lift_to_states(receiver_zone_density),
    lift_to_states(defenders_in_triangle_to_goal),
]

# silly_kicks/atomic/tracking/features.py
atomic_tracking_default_xfns = [
    lift_to_states(nearest_defender_distance),
    lift_to_states(actor_speed),
    lift_to_states(receiver_zone_density),
    lift_to_states(defenders_in_triangle_to_goal),
]
```

#### 4.2.6 Internal: `ActionFrameContext` and `_resolve_action_frame_context`

```python
@dataclasses.dataclass(frozen=True)
class ActionFrameContext:
    """Linkage + actor/opposite-team frame slices, computed once per add_action_context() call.
    Reused across the 4 feature kernels; future PR-S21+ features build their own kernels
    on this same Context shape."""
    actions: pd.DataFrame                       # subset of input actions (index aligned)
    pointers: pd.DataFrame                      # link_actions_to_frames output
    actor_rows: pd.DataFrame                    # one row per linked action: actor's frame row
    opposite_rows_per_action: pd.DataFrame      # long-form: action_id × opposite-team frame rows


def _resolve_action_frame_context(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
) -> ActionFrameContext:
    """Build the linked-context structure once. Internal — public API is via the
    per-feature helpers and add_action_context."""
```

### 4.3 Computational pattern

For all 4 features, the kernel is:

1. `_resolve_action_frame_context(actions, frames)` once per `add_action_context()` call (or once per per-feature helper call when called directly):
   - Calls `link_actions_to_frames(actions, frames)` → pointers DataFrame.
   - Inner-merges `pointers ⨝ frames on (period_id, frame_id)` → long-form (action × player) rows.
   - Splits into `actor_rows` (filter `frame.player_id == action.player_id`) and `opposite_rows_per_action` (filter `frame.team_id != action.team_id`).
2. Feature compute (vectorized):
   - `nearest_defender_distance`: cdist `(action.start_x, action.start_y)` ↔ `opposite_rows_per_action.x/y`, groupby `action_id` → min.
   - `actor_speed`: lookup `actor_rows.speed` indexed by `action_id`.
   - `receiver_zone_density`: cdist `(action.end_x, action.end_y)` ↔ `opposite_rows_per_action.x/y`, groupby `action_id` → count where `dist ≤ radius`.
   - `defenders_in_triangle_to_goal`: point-in-triangle test on `opposite_rows.x/y` with vertices `(action.start_x, action.start_y)`, `(105, 30.34)`, `(105, 37.66)` (vectorized barycentric / cross-product sign test); groupby `action_id` → sum.

All steps vectorized via pandas `merge` + numpy broadcasting + `groupby.agg`. No Cython/Numba needed at PR-S20 scale.

### 4.4 ADR-005 — Tracking-aware feature integration contract

Lives at `docs/superpowers/adrs/ADR-005-tracking-aware-features.md`. ~120-150 lines; mirrors ADR-001/002/003/004 shape. Captures the seven cross-cutting decisions:

1. **Frame-aware xfn protocol** — `_frame_aware = True` marker; `(states, frames) → Features` signature.
2. **VAEP base-class extension via composition, not subclass** — `compute_features` / `rate` gain optional `frames=None` kwarg; `is_frame_aware` dispatch. Rationale: avoid class-per-feature-kind explosion; AtomicVAEP gets parity free.
3. **Kernel-extraction pattern** — `silly_kicks/tracking/_kernels.py` as the schema-agnostic compute home. Sets precedent for PR-S21+ tracking features.
4. **Schema-aware-via-per-namespace-wrapper** — `silly_kicks.tracking.features` for standard SPADL, `silly_kicks.atomic.tracking.features` for atomic SPADL. Documents the choice over a schema-dispatcher alternative.
5. **LTR-normalization symmetric internal call** — VAEP normalizes both actions and frames internally, lazy-imports `tracking.utils.play_left_to_right` only when `frames is not None`. No vaep→tracking module-import-time coupling.
6. **Action-anchor for geometric features** — anchor on `action.start_x/y` (standard) or `action.x/y` (atomic), NOT on the actor's frame position. Rationale per spec §4.5; coordinate-alignment + VAEP-event-consistency.
7. **Linkage-provenance pass-through** — `add_action_context` joins `frame_id, time_offset_seconds, link_quality_score, n_candidate_frames` into the output. "Audit-by-default" convention.

### 4.5 Per-feature degradation policy (NaN semantics)

| Feature | Anchor | NaN cases | Zero cases |
|---|---|---|---|
| `nearest_defender_distance` | action.start_x/y | unlinked (frame_id NaN) | (none — distance is positive real) |
| `actor_speed` | actor's frame row | unlinked OR actor's player_id absent OR speed NaN | (none — speed is non-negative real) |
| `receiver_zone_density` | action.end_x/y, radius=5.0 | unlinked | linked but no defenders within radius (genuine count 0) |
| `defenders_in_triangle_to_goal` | action.start_x/y, goal posts | unlinked | linked but no defenders in wedge (genuine count 0) |

**Honest NaN with count-zero distinction:** the four features distinguish "data not available" (NaN, e.g., unlinked action) from "data available, count is genuinely zero" (0, e.g., no defenders in radius). Aligns with ADR-003 NaN-safety contract for `add_*` enrichment helpers.

**Coordinate-convention contract:** `add_action_context` requires `actions` and `frames` to use the same coordinate convention. Recommended: LTR-normalize both via `spadl.play_left_to_right` and `tracking.play_left_to_right` before calling. The function does not internally normalize when called directly (to preserve hexagonal pure-function discipline). Documented in the docstring; covered by a parity test that runs the function pre- and post-LTR-normalization on equivalent input and asserts identical output (modulo the LTR mirror).

When called via `VAEP.compute_features(frames=...)`, the VAEP method itself runs both `play_left_to_right` calls symmetrically before dispatching. So the typical user path (HybridVAEP) gets normalization for free.

### 4.6 Per-provider expected behaviour summary

Per the empirical-probe baselines (PR-S19's `tests/datasets/tracking/empirical_probe_baselines.json`):

| Provider | Frame rate | Linkage rate (synthetic) | actor_speed coverage | Geometric features coverage |
|---|---|---|---|---|
| PFF | 30 Hz | high | high (derived; NaN only on first frame of period or post-substitution) | high |
| Sportec | 25 Hz | high | high (native) | high |
| Metrica | 25 Hz | high | high (lakehouse provides native; library may derive — `speed_source` records origin) | high — note 77% NaN ball-coords on Metrica per probe; not a problem because we anchor on action coords |
| SkillCorner | 10 Hz | medium (broadcast) | high | high |

The 77% Metrica ball-NaN is the only per-provider asymmetry that comes near these features; the action-coord anchoring (§4.5) sidesteps it entirely.

## 5. Testing & validation

### 5.1 Test file layout

```
tests/
├── tracking/
│   ├── test_feature_framework.py            # frame_aware marker, is_frame_aware, ActionFrameContext, lift_to_states
│   ├── test_kernels.py                      # _kernels.py analytical-truth tests
│   ├── test_features_standard.py            # standard SPADL wrappers
│   ├── test_add_action_context.py           # aggregator + provenance + ADR-003 NaN safety
│   ├── test_action_context_cross_provider.py  # 4-provider parity (Tier 2 + 3)
│   ├── test_action_context_real_data_sweep.py # e2e-marked
│   └── (existing PR-S19 tests retained, untouched)
├── atomic/tracking/
│   ├── test_features_atomic.py              # atomic SPADL wrappers (parametrized over [standard, atomic])
│   └── test_atomic_action_context.py        # atomic aggregator
├── vaep/
│   ├── test_compute_features_frames_kwarg.py  # base.VAEP extension
│   ├── test_hybrid_with_tracking.py          # HybridVAEP integration + AUC test
│   └── (existing tests untouched)
├── atomic/vaep/
│   └── test_atomic_with_tracking.py         # AtomicVAEP integration test
├── test_todo_md_format.py                   # asserts On-Deck table shape (loose markdown check)
└── (existing test_enrichment_nan_safety.py auto-discovers add_action_context via @nan_safe_enrichment)

tests/datasets/tracking/
├── empirical_probe_baselines.json                    # PR-S19, retained
├── empirical_action_context_baselines.json           # NEW (Loop 0 output)
├── action_context_slim/                              # NEW Tier-3 lakehouse-derived
│   ├── sportec_slim.parquet
│   ├── metrica_slim.parquet
│   └── skillcorner_slim.parquet
└── (existing PR-S19 fixtures retained)
```

### 5.2 In-memory analytical fixtures (Tier 1)

Per-feature pytest fixtures with analytical ground truth:

- `nearest_defender_distance`: 3 defenders at known positions → expected min distance computable in test.
- `receiver_zone_density`: 5 defenders at radii [3, 4, 6, 8, 9] m, R=5.0 → expected count = 2.
- `defenders_in_triangle_to_goal`: defenders at known coords inside / outside the wedge.
- `actor_speed`: actor row in frames with known speed → exact match.

Each fixture is ~22 frame rows + 1-3 actions, built per-test in seconds.

### 5.3 Empirical-probe-driven baselines (Tier 3 + L3b)

`scripts/probe_action_context_baselines.py` — Loop 0, one-off, run during PR-S20 development. Output: `tests/datasets/tracking/empirical_action_context_baselines.json` (committed). Per-provider distribution stats (median, p25, p50, p75, p99) for the 4 features computed from real lakehouse Sportec/Metrica/SkillCorner + local PFF data. L3b sweep re-computes and asserts within tolerance.

Tier-3 lakehouse-derived parquets (~10 actions + linked frames per provider; ~2 KB each committed) reuse PR-S19's `_lakehouse_adapter.py`. Used by `test_action_context_cross_provider.py` for real-data cross-validation.

### 5.4 Cross-provider parity gate

```python
@pytest.mark.parametrize("provider", ["pff", "sportec", "metrica", "skillcorner"])
def test_action_context_bounds(provider):
    """All output rows pass: dist ≥ 0, speed ≥ 0, density int ≥ 0, triangle int ≥ 0."""

@pytest.mark.parametrize("provider", ["pff", "sportec", "metrica", "skillcorner"])
def test_action_context_link_rate(provider):
    """≥0.95 link rate on synthetic fixtures with action timestamps tied to real frame times."""

def test_action_context_distribution_overlap():
    """Per-provider feature percentiles overlap within tolerance — coordinate normalization
    consistent across providers (the schema-stress goal extended to action_context)."""
```

### 5.5 Real-data sweep (e2e-marked)

```python
@pytest.mark.e2e
@pytest.mark.parametrize("provider,env_var", [
    ("pff",         "PFF_TRACKING_DIR"),
    ("sportec",     "IDSSE_TRACKING_DIR"),
    ("metrica",     "METRICA_TRACKING_DIR"),
    ("skillcorner", "SKILLCORNER_TRACKING_DIR"),
])
def test_action_context_real_data_sweep(provider, env_var):
    """Per-provider full-match sweep:
       - Compute add_action_context on real actions + frames
       - Distribution match against empirical_action_context_baselines.json (within tolerance)
       - Bounds audit (no negative distances, no NaN in unexpected columns)
    """
```

`pytest.skip` with explicit reason on unset env var (memory: silently-skipping-tests-hide-breakage).

### 5.6 VAEP integration AUC test (Loop 9 + Loop 11)

```python
def test_hybrid_vaep_tracking_auc_uplift():
    """Train HybridVAEP twice on the same synthetic dataset:
       - baseline: xfns = hybrid_xfns_default
       - augmented: xfns = hybrid_xfns_default + tracking_default_xfns
       Assert AUC_augmented >= AUC_baseline + epsilon (epsilon=0.01 on synthetic).
       Proves: (1) the integration wires correctly end-to-end;
               (2) tracking features carry signal even on synthetic data."""

def test_atomic_vaep_tracking_smoke():
    """Mirrors the above on AtomicVAEP — smoke test (no AUC uplift assertion, just successful
    fit + rate cycle), since atomic synthetic AUC noise floor is higher."""
```

### 5.7 Public-API Examples coverage

Every public def in:
- `silly_kicks.tracking.feature_framework`
- `silly_kicks.tracking.features`
- `silly_kicks.atomic.tracking.features`

gets an `Examples` section. The existing `tests/test_public_api_examples.py` auto-discovers the new modules — no new test file.

Extended VAEP methods (`VAEP.compute_features`, `VAEP.rate` with new `frames` kwarg) get extended Examples sections in their docstrings showing the tracking-aware path.

### 5.8 Pre-commit gates

- `ruff` + `pyright` + `pytest -m "not e2e"` (Shift Left).
- `tests/test_action_context_real_data_sweep.py` run locally with all four `*_TRACKING_DIR` env vars set.
- `/final-review` (mandatory per `feedback_final_review_gate`).
- One commit per branch (squash merge).

### 5.9 TDD ordering (~13 RED-GREEN loops)

```
Loop 0 (one-off, scoped to PR-S20):
  - Write & run scripts/probe_action_context_baselines.py
  - Output: tests/datasets/tracking/empirical_action_context_baselines.json (committed)
  - Output: Tier-3 lakehouse-derived parquets in tests/datasets/tracking/action_context_slim/

Loop 1: feature_framework foundations
  - frame_aware decorator + is_frame_aware in silly_kicks/vaep/feature_framework.py
  - Frames type alias + FrameAwareTransformer
  - ActionFrameContext frozen dataclass + lift_to_states in silly_kicks/tracking/feature_framework.py

Loop 2: _resolve_action_frame_context (silly_kicks/tracking/utils.py extension)

Loop 3: lift_to_states tested end-to-end with stub feature

Loop 4: VAEP.compute_features extension (silly_kicks/vaep/base.py)
  - Backward-compat regression test (frames=None unchanged)
  - Frame-aware dispatch
  - ValueError on missing frames

Loop 5: VAEP.rate extension (silly_kicks/vaep/base.py)

Loop 6: _kernels.py — pure compute kernels with analytical Tier-1 fixtures

Loop 7: tracking.features.py wrappers (standard SPADL)

Loop 8: add_action_context aggregator + provenance columns + @nan_safe_enrichment

Loop 9: tracking_default_xfns + HybridVAEP integration AUC test

Loop 10: atomic.tracking.features.py wrappers (parametrized over standard/atomic)

Loop 11: AtomicVAEP integration smoke test

Loop 12: TODO.md restructure to lakehouse format + ADR-005 authoring + NOTICE
         file creation (academic-attribution canonical record per §9) + README.md
         and CLAUDE.md cross-links + tests/test_notice_md_format.py structural check

Loop 13: Cross-provider parity (L2b) + real-data sweep (L3b) + final-review

Post-loops:
  - /final-review (mandatory)
  - Single commit, squash merge
```

## 6. Branch / version strategy

- **Branch:** `feat/tracking-action-context-pr1`.
- **Target version:** silly-kicks **2.8.0** (minor bump — additive feature set, no API breakage in any existing module).
- **Single commit per branch** (memory: commit policy). Squash-merge on GitHub.
- **PR-S21** (`pre_shot_gk_position_*`, separate session) targets **2.8.x** or **2.9.0** depending on whether PR-S21 stacks on the same TODO/On-Deck bench or absorbs other GK-coach work.

## 7. Lakehouse boundary impact (informational)

PR-S20 adds 4 columns (+4 provenance columns) to `add_action_context` output but does NOT modify any lakehouse mart. After PR-S20 ships, the lakehouse-side work is:

1. New silly-kicks consumer cell that calls `add_action_context(actions, frames)` per match.
2. New columns in `fct_action_values` (or a new `fct_action_features_tracking_v1` mart) wired to the silly-kicks output.
3. Wire `tracking_default_xfns` into the lakehouse's HybridVAEP training cell for next-cycle xG/VAEP calibration improvement.

The lakehouse-internal probe (2026-04-30, this spec) confirmed the lakehouse does NOT precompute these 4 features in any existing mart. PR-S20 is the canonical implementation; the lakehouse will consume.

The `fct_defcon_pressure` mart (9,826 rows; columns `total_pressure`, `intercept_pressure`, `concede_pressure`, `disturb_pressure`, `deter_pressure`) is the formula reference for PR-S22 (TF-2, `pressure_on_actor`), not PR-S20.

## 8. TODO.md restructure

**New file shape** (mirroring `karstenskyt__luxury-lakehouse/TODO.md`):

```markdown
# silly-kicks — TODO

Quick-reference action items. Architectural decisions live in [docs/superpowers/adrs/](docs/superpowers/adrs/).

**Last updated**: 2026-04-30 (PR-S20 cycle in flight — `feat/tracking-action-context-pr1`)
**(A) silly-kicks 2.7.0 SHIPPED** (PR-S19 — tracking namespace primitive layer; ADR-004).

---

## On Deck

| Size | What it means |
|------|---------------|
| **Monstah** | Multi-phase epic |
| **Wicked** | Looks small, surprisingly impactful |
| **Dunkin'** | Quick run, keeps things moving |

| # | Task | Size | Source | Notes |
|---|------|------|--------|-------|
| TF-1 | `pre_shot_gk_position_*` refining `add_pre_shot_gk_context` | Dunkin' | Anzer & Bauer 2021 (xG with synchronized tracking); Bekkers 2024 (DEFCON); ADR-004 deferred | **GK-coach priority pickup.** Replace heuristic GK position estimation with linked-frame GK x/y. **References:** Anzer, G., & Bauer, P. (2021), "A goal scoring probability model for shots based on synchronized positional and event data in football and futsal." Frontiers in Sports and Active Living, 3, 624475 — uses GK position as a key xG feature. ~50-100 LOC + tests. |
| TF-2 | `pressure_on_actor()` — pressure feature with formula choice | Wicked | Andrienko et al. 2017; Link et al. 2016; Bekkers 2024; lakehouse `fct_defcon_pressure` (9,826 rows formula reference) | **Formula-choice spec decision.** **References:** Andrienko, G., Andrienko, N., Budziak, G., Dykes, J., Fuchs, G., von Landesberger, T., & Weber, H. (2017), "Visual analysis of pressure in football." Data Mining and Knowledge Discovery, 31, 1793-1839 (cone-sum 1/d^k). Link, D., Lang, S., & Seidenschwarz, P. (2016), "Real Time Quantification of Dangerousity in Football Using Spatiotemporal Tracking Data." PLOS ONE, 11(12) (exponential decay). Bekkers, J. (2024), DEFCON-style (distance × angle). Inherit whichever formula the lakehouse `fct_defcon_pressure` mart implements (probe at PR-S22 start to confirm which) for cross-pipeline parity. ~150 LOC + possible ADR-006 if formula choice is contentious. |
| TF-3 | `actor_distance_pre_window` — 0.5 s pre-action distance traveled | Dunkin' | Bauer & Anzer 2021 (counterpressing detection); New PR-S20 deferral | **References:** Bauer, P., & Anzer, G. (2021), "Data-driven detection of counterpressing in professional football." Data Mining and Knowledge Discovery, 35(5), 2009-2049 — uses pre-action movement as feature. First time-window feature; uses `slice_around_event` (PR-S19). ~80 LOC. |
| TF-4 | `off_ball_runs` — attacking teammate runs in pre-action window | Wicked | Power et al. 2017 (OBSO); Spearman 2018; Decroos & Davis 2020 | **References:** Power, P., Ruiz, H., Wei, X., & Lucey, P. (2017), "Not all passes are created equal." KDD '17 (OBSO — Off-Ball Scoring Opportunity). Spearman, W. (2018), "Beyond Expected Goals." MIT Sloan SAC. Decroos, T., & Davis, J. (2020), Player-Vectors blog/extension of VAEP. Per-teammate temporal analysis. Heaviest of the deferred bench. ~200-300 LOC + scaling considerations. |
| TF-5 | `infer_ball_carrier(frames, tolerance_m=...)` | Wicked | Lakehouse session (2026-04-30); Bauer & Anzer 2021 (uses ball-carrier identification implicitly); ADR-004 #3 | Heuristic per-frame carrier inference (closest-player-to-ball-with-velocity-toward-ball). No single canonical academic reference — most papers assume the carrier is given. **Pragmatic reference:** Bauer, P., & Anzer, G. (2021), Section 3 — describes carrier-identification heuristic similar to ours. Foundational utility; many downstream features will consume. ~150 LOC. |
| TF-6 | `sync_score` per-action tracking↔events sync-quality score | Dunkin' | ADR-004 #4; novel utility (no canonical academic reference) | QA primitive. Reuses `link_actions_to_frames` pointers + `link_quality_score`. No academic citation required — this is library-internal QA, not a published metric. ~50 LOC. |
| TF-7 | Pitch-control models (Spearman / Voronoi) | Monstah | Spearman 2018; Fernández & Bornn 2018; Spearman et al. 2017; ADR-004 #5 | **References:** Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017), "Physics-Based Modeling of Pass Probabilities in Soccer." MIT Sloan SAC. Spearman, W. (2018), "Beyond Expected Goals." MIT Sloan SAC. Fernández, J., & Bornn, L. (2018), "Wide Open Spaces: A statistical technique for measuring space creation in professional soccer." MIT Sloan SAC. Numba acceleration, broadcast-tracking edge cases, validation harness. Own scoping cycle. |
| TF-8 | Smoothing primitives (Savitzky-Golay, EMA) | Dunkin' | Savitzky & Golay 1964 (canonical); ADR-004 #6 | **References:** Savitzky, A., & Golay, M. J. E. (1964), "Smoothing and Differentiation of Data by Simplified Least Squares Procedures." Analytical Chemistry, 36(8), 1627-1639. Per-provider preprocessor. ~80 LOC. |
| TF-9 | Multi-frame interpolation / gap filling | Dunkin' | Standard numerical methods (no domain-specific paper); ADR-004 #7 | Standard cubic-spline / linear interpolation. No domain-specific citation; standard signal-processing practice. ~100 LOC. |
| TF-10 | Lakehouse boundary adapter for `add_action_context` outputs | Wicked | Lakehouse-side; tracked here for cross-repo visibility | Wires PR-S20's 4 features into `fct_action_values` / new mart. Not in silly-kicks repo; logged for coordination. References inherited from PR-S20 spec §11. |

---

## Active Cycle

PR-S20 — `action_context()` tracking-aware features (target silly-kicks 2.8.0).

Branch: `feat/tracking-action-context-pr1`. Spec + plan: [docs/superpowers/specs/2026-04-30-action-context-pr1-design.md](docs/superpowers/specs/2026-04-30-action-context-pr1-design.md), [docs/superpowers/plans/2026-04-30-action-context-pr1.md](docs/superpowers/plans/2026-04-30-action-context-pr1.md).

After ship, this section gets archived; ADR-005 staged-rollout (if any) becomes the durable record.

---

## Technical Debt

### Blocked or Deferred

(none currently queued)

---

## Research & Future Work

ReSpo.Vision tracking adapter — licensing-blocked. Track here when licensing clears.
```

(Existing legacy sections "Documentation / Architecture / Open PRs / Tech Debt" — empty or near-empty in current TODO.md — fold into the new structure or remove if empty.)

## 9. NOTICE file (academic-attribution canonical record)

silly-kicks does not currently have a `NOTICE` file at the repo root. PR-S20 establishes one, mirroring the lakehouse pattern (`karstenskyt__luxury-lakehouse/NOTICE`).

### 9.1 Why now

PR-S20 ships the first features whose academic methodologies are *external* to the foundational VAEP/SPADL/xT body that silly-kicks already inherits from socceraction. The four `action_context()` features draw from Lucey 2014, Anzer & Bauer 2021, Spearman 2018, Power 2017, and Pollard & Reep 1997 — none of which is captured in the current `LICENSE` (which credits only the upstream KU Leuven authors). Without a NOTICE file:

- New academic methodologies have no canonical home for attribution.
- Per-source-file `# References` comments fragment over time.
- Future PRs (TF-1 GK refinement, TF-2 pressure with three competing formulas, TF-7 pitch-control) compound the gap.

PR-S20 is the right time because (a) the academic-attribution discipline is already being established for this PR, (b) the NOTICE file is small to write now and grows additively, (c) it matches the lakehouse pattern the user explicitly invoked.

### 9.2 NOTICE file shape (mirrors `luxury-lakehouse/NOTICE`)

```
silly-kicks
Copyright (c) 2019 KU Leuven Machine Learning Research Group (Tom Decroos, Pieter Robberechts)
Copyright (c) 2026 Karsten S. Nielsen

This product is a maintained fork of socceraction
(https://github.com/ML-KULeuven/socceraction). Major architectural changes
since 1.0.0 are documented in CHANGELOG.md.

Third-Party Libraries
---------------------

kloppy --- standardizing soccer tracking/event data (BSD-3-Clause License).
Copyright (c) kloppy contributors.
See: https://github.com/PySport/kloppy

pandas, numpy, scikit-learn --- core dependencies (BSD / standard licenses).

Optional gradient-boosting backends (xgboost, lightgbm, catboost) --- listed
in pyproject.toml; each retains its upstream license.

Mathematical / Methodological References
----------------------------------------

The SPADL action representation (silly_kicks/spadl/) implements the framework
described in: Decroos, T., Van Haaren, J., & Davis, J. (2018). "SPADL: A
Common Framework for Action Description in Soccer." Workshop on Machine
Learning and Data Mining for Sports Analytics (ECML-PKDD).

The VAEP action valuation framework (silly_kicks/vaep/) implements:
Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). "Actions
Speak Louder Than Goals: Valuing Player Actions in Soccer." Proc. KDD '19.
The HybridVAEP variant (silly_kicks/vaep/hybrid.py) is a result-leakage-
removal variant introduced in this fork; no separate academic citation.

The Atomic-SPADL representation and Atomic-VAEP framework
(silly_kicks/atomic/) implement: Decroos, T., Robberechts, P., & Davis, J.
(2020). "Introducing Atomic-SPADL: A New Way to Represent Event Stream Data."
DTAI Sports Analytics Blog.

The Expected Threat (xT) grid (silly_kicks/xthreat.py) seeds from:
Singh, K. (2018). "Introducing Expected Threat (xT)." karun.in/blog/expected-threat
The grid is recomputable from event data; the seed values are reference-only.

The tracking namespace primitive layer (silly_kicks/tracking/, PR-S19,
ADR-004) implements ingestion + linkage primitives across PFF, Sportec,
Metrica, and SkillCorner. No new academic methodology beyond the canonical
ADR.

The four tracking-aware action-context features in
silly_kicks/tracking/features.py and silly_kicks/atomic/tracking/features.py
(PR-S20, ADR-005) implement methodologies described in:

- Lucey, P., Bialkowski, A., Monfort, M., Carr, P., & Matthews, I. (2014).
  "Quality vs Quantity: Improved Shot Prediction in Soccer using Strategic
  Features from Spatiotemporal Data." MIT Sloan Sports Analytics Conference.
  (canonical "defenders in shot triangle" feature; nearest-defender-distance
  for shots)

- Anzer, G., & Bauer, P. (2021). "A goal scoring probability model for shots
  based on synchronized positional and event data in football and futsal."
  Frontiers in Sports and Active Living, 3, 624475.
  (player_speed and distance-to-defender as xG features)

- Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan Sports Analytics
  Conference.
  (zone-based defender intensity in pitch-control framework)

- Power, P., Ruiz, H., Wei, X., & Lucey, P. (2017). "Not all passes are
  created equal: Objectively measuring the risk and reward of passes in
  soccer from tracking data." KDD '17 (OBSO).
  (receiver-zone risk/reward modelling)

- Pollard, R., & Reep, C. (1997). "Measuring the effectiveness of playing
  strategies at soccer." Journal of the Royal Statistical Society Series D,
  46(4), 541-550.
  (early shot-quality / pressure-from-defenders concept)

The implementations are independent Python translations of the published
methodologies, not derived from any source code. Licensed under the same
terms as silly-kicks (MIT License).
```

### 9.3 Cross-linking

- **`README.md`** gains an "Attribution" section pointing at `NOTICE` near the bottom (additive, ~3 lines).
- **`CLAUDE.md`** gains one line under conventions: `Academic attribution: every new feature with a published methodology gets an entry in the NOTICE file's "Mathematical / Methodological References" section. Cross-link from per-feature docstrings.`
- Per-feature docstrings (the `References` sections written for the 4 features in §4.2.1) include a one-line cross-reference: `See NOTICE for full bibliographic citations.` — so individual function readers know where the canonical record lives.

### 9.4 Maintenance discipline (PR-S21+)

Every future PR shipping a new feature with a published methodology adds an entry to the NOTICE file's "Mathematical / Methodological References" section. The TF-1..TF-9 entries in TODO.md already carry the references that will populate those NOTICE entries when shipped — so the maintenance contract is "TODO Source/Notes column → NOTICE entry on ship".

A lightweight CI test (`tests/test_notice_md_format.py`, ~30 LOC) asserts:
- `NOTICE` file exists at repo root.
- README.md has a hyperlink to NOTICE.
- A loose markdown-structure check (presence of "Mathematical / Methodological References" section).
- One regression case: each of the five PR-S20 references is present in NOTICE by author surname (Lucey, Anzer, Spearman, Power, Pollard).

This catches accidental drift in low-cost, structural ways. NOT enforced: that every public function with a `References` docstring has a corresponding NOTICE entry — too brittle, would surface stylistic flaps and false positives.

## 10. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| `VAEP.compute_features` extension breaks existing callers | Backward-compat regression test in Loop 4 asserts `compute_features(frames=None)` is bit-identical to today. Default kwarg preserves every existing call site. |
| Lazy `from silly_kicks.tracking.utils import play_left_to_right` inside vaep.base creates import cycle | Tracking imports vaep.feature_framework (for the marker); vaep.base lazy-imports tracking only when frames is not None. Loop 4 includes a test that imports vaep.base alone (without tracking) to confirm no cycle. |
| HybridVAEP integration AUC uplift fails on synthetic data | Loop 9 fixture is constructed so tracking features have signal (e.g., shots near defenders have lower probability of scoring). If signal is too weak on truly-synthetic data, the AUC test ε is loosened OR the fixture is regenerated with stronger tracking-correlated outcomes (user-authorized regeneration per session policy). |
| Atomic-SPADL feature semantics differ unexpectedly | Loop 10 parametrizes per-feature tests over `[(standard, atomic)]`, surfacing semantic divergence early. Atomic-specific edge case (`dx == dy == 0` actions) documented in spec §5 + tested. |
| Coordinate-convention mismatch between actions and frames silently wrong | Documented in `add_action_context` docstring + tested via parity test that runs pre- and post-LTR-normalization. VAEP integration auto-normalizes on the typical path. |
| Lakehouse-derived Tier-3 parquets drift as lakehouse evolves | L3b sweep re-computes baselines from local data; deviation surfaces before merge. Same drift-detection pattern as PR-S19. |
| Performance at scale (multi-season runs) | Benchmark in Loop 6 confirms vectorized pandas is sufficient at PR-S20 scale. If a real-world bottleneck surfaces post-merge, document on On-Deck as TF-* and revisit; don't pre-optimize. |

## 11. References

### Internal — silly-kicks specs / ADRs / data baselines

- `docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md` — PR-S19 predecessor.
- `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md` — identifier convention precedent.
- `docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md` — NaN-safety contract for `add_*` helpers.
- `docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md` — the 9 invariants PR-S20 inherits.
- `docs/superpowers/adrs/ADR-005-tracking-aware-features.md` — NEW; codifies the 7 cross-cutting decisions in this spec.
- `tests/datasets/tracking/empirical_probe_baselines.json` — PR-S19 baselines (retained).
- `tests/datasets/tracking/empirical_action_context_baselines.json` — PR-S20 baselines (NEW, Loop 0).
- Lakehouse cross-check session (2026-04-30) — recommended impact-bundle of nearest-defender distance + ball-carrier speed + receiver-zone density as priority #1.
- Lakehouse action-feature mart probe (2026-04-30, this spec): confirmed no existing precomputation of the 4 PR-S20 features; lakehouse will be a consumer.

### Academic literature directly informing PR-S20 features

- **Decroos, T., Bransen, L., Van Haaren, J., & Davis, J.** (2019). "Actions Speak Louder Than Goals: Valuing Player Actions in Soccer." *Proc. KDD '19*, 1851-1861. — Foundational VAEP paper; Hybrid VAEP variant (this repo) builds on this.
- **Decroos, T., & Davis, J.** (2020). "Player Vectors and Tracking-Aware VAEP." DTAI Sports Analytics blog / extension. — Documents the impact of nearest-defender-distance and similar tracking features on VAEP calibration.
- **Lucey, P., Bialkowski, A., Monfort, M., Carr, P., & Matthews, I.** (2014). "Quality vs Quantity: Improved Shot Prediction in Soccer using Strategic Features from Spatiotemporal Data." *MIT Sloan Sports Analytics Conference*. — Canonical "defenders in shot triangle" feature; nearest-defender-distance for shots.
- **Anzer, G., & Bauer, P.** (2021). "A goal scoring probability model for shots based on synchronized positional and event data in football and futsal." *Frontiers in Sports and Active Living*, 3, 624475. — Tracking-aware xG with player_speed, distance-to-defender, GK-position features. Direct reference for `nearest_defender_distance` and `actor_speed`.
- **Spearman, W.** (2018). "Beyond Expected Goals." *MIT Sloan Sports Analytics Conference*. — Pitch-control framework using receiver-zone defender intensity. Direct reference for `receiver_zone_density`.
- **Power, P., Ruiz, H., Wei, X., & Lucey, P.** (2017). "Not all passes are created equal: Objectively measuring the risk and reward of passes in soccer from tracking data." *Proc. KDD '17* (OBSO). — Receiver-zone risk/reward modelling; reference for `receiver_zone_density` and PR-S21+ off-ball-runs.
- **Pollard, R., & Reep, C.** (1997). "Measuring the effectiveness of playing strategies at soccer." *Journal of the Royal Statistical Society Series D*, 46(4), 541-550. — Early shot-quality / pressure-from-defenders concept; antecedent of the shot-triangle feature.

### Academic literature for deferred TF-* items

(Cited in TODO.md On-Deck Notes; collected here for the spec's comprehensive bibliography.)

- **Bauer, P., & Anzer, G.** (2021). "Data-driven detection of counterpressing in professional football: A supervised machine learning task based on synchronized positional and event data with expert-based feature extraction." *Data Mining and Knowledge Discovery*, 35(5), 2009-2049. — Pre-action movement features (TF-3); ball-carrier-identification heuristic (TF-5).
- **Bekkers, J.** (2024). *DEFCON-style pressure metrics from tracking data*. — Distance×angle pressure formula (TF-2 candidate).
- **Andrienko, G., Andrienko, N., Budziak, G., Dykes, J., Fuchs, G., von Landesberger, T., & Weber, H.** (2017). "Visual analysis of pressure in football." *Data Mining and Knowledge Discovery*, 31, 1793-1839. — Cone-sum 1/d^k pressure formula (TF-2 candidate).
- **Link, D., Lang, S., & Seidenschwarz, P.** (2016). "Real Time Quantification of Dangerousity in Football Using Spatiotemporal Tracking Data." *PLOS ONE*, 11(12), e0168768. — Exponential-decay pressure formula (TF-2 candidate).
- **Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P.** (2017). "Physics-Based Modeling of Pass Probabilities in Soccer." *MIT Sloan Sports Analytics Conference*. — Pitch-control physics model (TF-7).
- **Fernández, J., & Bornn, L.** (2018). "Wide Open Spaces: A statistical technique for measuring space creation in professional soccer." *MIT Sloan Sports Analytics Conference*. — Pitch-control alternative (TF-7).
- **Savitzky, A., & Golay, M. J. E.** (1964). "Smoothing and Differentiation of Data by Simplified Least Squares Procedures." *Analytical Chemistry*, 36(8), 1627-1639. — Smoothing primitive (TF-8).
