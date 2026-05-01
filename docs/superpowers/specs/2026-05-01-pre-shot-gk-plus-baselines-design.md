# Pre-shot GK position + baselines backfill — PR-S21 — silly-kicks 2.9.0

**Status:** Approved (design)
**Target release:** silly-kicks 2.9.0
**Author:** Karsten S. Nielsen with Claude Opus 4.7 (1M)
**Date:** 2026-05-01
**Predecessor:** silly-kicks 2.8.0 — PR-S20 (`action_context()` tracking-aware features; ADR-005)
**Successors:** TF-12 (`pre_shot_gk_angle_*`), TF-13 (frame-based GK identification fallback), TF-14 (defensive-line features) — bundled as TODO.md additions in this PR per the National Park Principle.

**Triggers:**

- TODO.md On-Deck row TF-1: `pre_shot_gk_position_*` refining `add_pre_shot_gk_context` (Dunkin'; GK-coach priority pickup).
- TODO.md On-Deck row TF-11: backfill PR-S20 distribution baselines into `empirical_action_context_baselines.json` (Dunkin'; PR-S20 forward-reference).
- ADR-005's 7 cross-cutting decisions are the contract; PR-S21 ships entirely within the locked envelope (no new ADR).
- Best-practice, long-term TDD/E2E discipline: full atomic-SPADL parity from day one; bit-exact per-row regression gate (no statistical-noise tolerance).
- National Park Principle: bundle 3 new GK / defensive-line TODO additions surfaced during scoping into the same commit so they don't get lost.

---

## 1. Problem

silly-kicks 2.8.0 (PR-S20) shipped 4 tracking-aware action-context features with a locked integration contract (ADR-005). Two adjacent items remain on the On-Deck table from that cycle:

1. **TF-1 — `pre_shot_gk_position_*`.** The events-side `add_pre_shot_gk_context` helper currently emits 4 columns about defending-GK *engagement state* (`gk_was_distributing`, `gk_was_engaged`, `gk_actions_in_possession`, `defending_gk_player_id`). It does NOT estimate any GK *position* — callers wanting GK x/y for xG / xGOT modeling currently have to assume the GK is on the goal line at e.g. (105, 34). TF-1 supplies the GK's actual x/y and 2 derived distances from the linked frame, when tracking is supplied.

2. **TF-11 — baselines backfill.** PR-S20 committed `empirical_action_context_baselines.json` with provenance keys + 64 null percentile slots (4 features × 4 percentiles × 4 providers). Filling them was deferred. Concurrently, PR-S20's cross-provider parity test only verifies bounds + linkage rate — there is no bit-exact regression gate against the committed slim parquets, so kernel drift can land silently.

The lakehouse 2026-04-30 cross-check session ranked the GK-position feature as a GK-coach priority pickup (Anzer & Bauer 2021 use GK position as a key xG feature; the lakehouse does not precompute it today). The two items are bundled into one PR because both are Dunkin'-sized and touch the same general tracking-aware GK / context-features area; they share a regenerator script.

This PR (PR-S21) lands TF-1 + TF-11 under ADR-005's locked architecture, plus 3 bundled TODO.md additions (TF-12 / TF-13 / TF-14) for follow-up GK / defensive-line work.

## 2. Goals

1. **4-feature GK-position catalog**, all anchored on the linked frame's defending-GK row:
   - `pre_shot_gk_x` — float64, m; GK's x at linked frame, LTR-normalized.
   - `pre_shot_gk_y` — float64, m; GK's y at linked frame, LTR-normalized.
   - `pre_shot_gk_distance_to_goal` — float64, m; Euclidean to goal-mouth center (105, 34).
   - `pre_shot_gk_distance_to_shot` — float64, m; Euclidean to shot anchor (`action.start_x/y` standard; `action.x/y` atomic).
2. **Two public surfaces** (Fork 1 resolution = C):
   - Tracking-namespace canonical compute: `silly_kicks.tracking.features.pre_shot_gk_x/_y/_distance_to_goal/_distance_to_shot` per-feature Series helpers + `add_pre_shot_gk_position` aggregator + `pre_shot_gk_default_xfns` lifted-xfn list.
   - Events-side wrapper extension: `add_pre_shot_gk_context(actions, *, frames=None, ...)` gains optional `frames` kwarg; when supplied, lazy-imports the canonical compute and merges 4 GK-position columns + 4 provenance columns into the output. When `frames=None`, behavior is bit-identical to silly-kicks 2.8.0.
3. **Full atomic-SPADL parity** at PR-S21 ship time. Standard SPADL and Atomic SPADL each get their own `tracking.features` GK helpers sharing a single private kernel (`silly_kicks/tracking/_kernels.py`).
4. **VAEP / HybridVAEP / AtomicVAEP integration** via composition. `pre_shot_gk_default_xfns` is a separate list (not appended into `tracking_default_xfns`) — composability + GK is shot-only signal; mixing it into the universal default forces the `defending_gk_player_id` requirement on all PR-S20 callers (OI-3 resolution).
5. **TF-11 hybrid validation strategy** (Fork 2 resolution):
   - **Per-row regression gate (load-bearing)**: `*_expected.parquet` per provider; `pd.testing.assert_frame_equal(atol=1e-9, rtol=0)`. Bit-exact, deterministic, no statistical noise. Failure mode is "row 5: expected 4.32, got 4.51" — fully debuggable.
   - **JSON baselines as documentation**: 64 null slots backfilled. Role: human-readable distribution summary committed alongside the data + JSON-vs-parquet consistency assertion. NOT the per-feature drift gate.
   - **Cross-provider distribution overlap test** (existing PR-S20) is the genuinely valuable distribution-level check; unchanged.
6. **TDD-first**, RED-before-GREEN. Test coverage: kernel analytical truth (Tier 1), schema-wrapper parity (Tier 2), per-row regression on slim parquets (Tier 3, TF-11), e2e real-data sweep (4 providers).
7. **NOTICE attribution discipline**: extend the existing Anzer & Bauer (2021) bullet to enumerate the GK-position feature (Fork 3 resolution = extend, not separate bullet).
8. **Backward-compat-bit-identity** for `add_pre_shot_gk_context(actions)` (no frames kwarg). Pinned by golden-fixture test.
9. **National Park bundle**: TF-12 / TF-13 / TF-14 added to TODO.md On-Deck table as part of PR-S21 commit.

## 3. Non-goals

- **No GK angle features.** Multiple competing conventions (relative to shot trajectory? to goal-line normal? signed vs unsigned?). Library ships positions + 2 unambiguous distances; downstream computes angles in their preferred convention. TF-12 logged for follow-up.
- **No frame-based defending-GK identification.** PR-S21 strictly uses events-based `defending_gk_player_id` (set by the events-only step inside `add_pre_shot_gk_context`). When that ID is NaN (no defending-keeper engagement in the lookback window), GK-position output is NaN — honest about pre-engagement shots, no fabrication. TF-13 logged as Wicked-sized follow-up with own academic literature.
- **No pressure-on-keeper, GK-off-line-time-window, or defensive-line features.** Generic pressure is TF-2; defensive-line features are TF-14 (logged in this PR). PR-S21 does not ship them.
- **No new tracking provider adapters.** PR-S21 consumes the 4-provider coverage from PR-S19; ReSpo.Vision remains licensing-blocked.
- **No new ADR.** PR-S21 ships entirely within ADR-005's 7 cross-cutting decisions. The `ActionFrameContext` field addition is a non-breaking extension conforming to ADR-005 §3.
- **No lakehouse boundary adapter.** Lakehouse pipelines build their own boundary in their own repo. TF-10 (PR-S20) tracks this.
- **No CHANGELOG handwriting in this session.** Spec session writes the spec + plan only; CHANGELOG is a Loop-N task in the implementation plan.
- **No streaming / chunked processing.** Per ADR-004 invariant 1 (hexagonal pure-function contract).
- **No new wide-form / 120×80 / object-id schema variants.** PR-S21 inherits PR-S19's locked schema verbatim.

## 4. Design

### 4.1 Architecture & module layout

```
silly_kicks/tracking/
├── _kernels.py                  # extend: _pre_shot_gk_position kernel
├── feature_framework.py         # extend: ActionFrameContext.defending_gk_rows: pd.DataFrame
├── features.py                  # extend: 4 GK Series helpers + add_pre_shot_gk_position + pre_shot_gk_default_xfns
├── utils.py                     # extend: _resolve_action_frame_context populates defending_gk_rows
└── (existing schema / sportec / pff / kloppy / _direction unchanged)

silly_kicks/atomic/tracking/
└── features.py                  # mirror: same 4 helpers + aggregator + xfn list

silly_kicks/spadl/utils.py       # add_pre_shot_gk_context gains optional frames=None kwarg + 4 GK columns + provenance
silly_kicks/atomic/spadl/utils.py # mirror

scripts/
└── regenerate_action_context_baselines.py   # NEW — TF-11 regenerator (parquets + JSON)

NOTICE                                       # extend Anzer & Bauer bullet
TODO.md                                      # mark TF-1 + TF-11 SHIPPED + add TF-12/13/14 (National Park bundle)
docs/superpowers/specs/2026-05-01-pre-shot-gk-plus-baselines-design.md   # this spec
docs/superpowers/plans/2026-05-01-pre-shot-gk-plus-baselines.md          # plan (next session)
```

**Direction-of-dependency.** `spadl.utils → tracking.features` is a NEW edge but lazy-imported inside the `if frames is not None` branch — preserving ADR-005 §5 (no module-import-time cycle). Pyright sees the type via `if TYPE_CHECKING:` import; runtime call site lazy-imports. Pattern ports verbatim from `vaep.base` (PR-S20 Loop 4).

**Single source of truth for GK identity.** The events-side step (recent-`keeper_*` lookup) determines `defending_gk_player_id` first; the tracking-namespace primitives REQUIRE that column to be present in `actions` and raise `ValueError` if missing. No new GK-identification heuristic introduced. NaN `defending_gk_player_id` → NaN GK-position output (honest, no fabrication).

### 4.2 Public API

#### 4.2.1 Tracking-namespace surface (`silly_kicks/tracking/features.py`)

```python
def pre_shot_gk_x(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series: ...
def pre_shot_gk_y(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series: ...
def pre_shot_gk_distance_to_goal(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series: ...
def pre_shot_gk_distance_to_shot(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.Series: ...

@nan_safe_enrichment
def add_pre_shot_gk_position(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Enrich actions with 4 GK-position columns + 4 linkage-provenance columns.

    REQUIRES `defending_gk_player_id` column in actions
    (run silly_kicks.spadl.utils.add_pre_shot_gk_context first).
    Non-shot rows / unlinked rows / pre-engagement rows / GK-absent-from-frame rows get NaN.
    """

pre_shot_gk_default_xfns = [
    lift_to_states(pre_shot_gk_x),
    lift_to_states(pre_shot_gk_y),
    lift_to_states(pre_shot_gk_distance_to_goal),
    lift_to_states(pre_shot_gk_distance_to_shot),
]
```

Atomic mirror has identical 4 + aggregator + xfn list, calling the same `_kernels._pre_shot_gk_position` with anchor `(actions["x"], actions["y"])` and atomic shot type ids `{"shot", "shot_penalty"}`.

#### 4.2.2 Events-side wrapper (standard `silly_kicks/spadl/utils.py`, atomic mirror)

```python
@nan_safe_enrichment
def add_pre_shot_gk_context(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame | None = None,    # NEW
    lookback_seconds: float = 10.0,
    lookback_actions: int = 5,
) -> pd.DataFrame:
    """... [existing docstring, extended] ...

    When ``frames`` is supplied, additionally emits 4 GK-position columns
    (pre_shot_gk_x, pre_shot_gk_y, pre_shot_gk_distance_to_goal,
    pre_shot_gk_distance_to_shot) + 4 linkage-provenance columns
    (frame_id, time_offset_seconds, link_quality_score, n_candidate_frames)
    via the silly_kicks.tracking.features canonical compute. When
    frames=None (default), behavior is bit-identical to silly-kicks 2.8.0
    — no frames-related columns appear in the output.
    """
```

Internally: events-only step runs first (populating `defending_gk_player_id` + 4 engagement columns). If `frames is not None`, lazy-imports `silly_kicks.tracking.features.add_pre_shot_gk_position` and merges its 4 GK-position columns + 4 provenance columns into the output.

**Backward-compat:** every existing caller (`add_pre_shot_gk_context(actions)` positional or no-kwarg) gets bit-identical output. Pinned by golden-fixture test.

### 4.3 Data flow & kernel

**`ActionFrameContext` extension** (`silly_kicks/tracking/feature_framework.py`):

```python
@dataclasses.dataclass(frozen=True)
class ActionFrameContext:
    actions: pd.DataFrame
    pointers: pd.DataFrame
    actor_rows: pd.DataFrame
    opposite_rows_per_action: pd.DataFrame
    defending_gk_rows: pd.DataFrame    # NEW — empty DataFrame when defending_gk_player_id absent from actions
```

`defending_gk_rows` is long-form: one row per (linked action × frame row where `frame.player_id == action.defending_gk_player_id` AND `not is_ball`). Empty when `defending_gk_player_id` is absent / NaN, the action is unlinked, or the GK player_id is absent from the linked frame (substitution case).

**`_resolve_action_frame_context` extension** (`silly_kicks/tracking/utils.py`): the existing `actions_with_period` projection is widened to also pull `defending_gk_player_id` if present. After the join, a new mask filters rows where `player_id_frame == defending_gk_player_id` (and `not is_ball`); both sides cast to float64 to dodge dtype-mismatch warnings (R3 mitigation).

**Kernel** (`silly_kicks/tracking/_kernels.py`), schema-agnostic per ADR-005 §3:

```python
def _pre_shot_gk_position(
    anchor_x: pd.Series,                # action.start_x (standard) or action.x (atomic)
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
    *,
    shot_type_ids: frozenset[int],      # caller passes the right set per schema
) -> pd.DataFrame:
    """Returns DataFrame indexed by ctx.actions.index with 4 columns.
    All NaN for non-shots / unlinked / pre-engagement / GK-absent rows."""
```

Compute (vectorized, ~30 LOC):
1. Build per-action GK row by left-joining `defending_gk_rows[[action_id, x, y]]` onto a per-action shot mask (`type_id ∈ shot_type_ids`). Non-shot / unlinked / pre-engagement / GK-absent → all NaN.
2. `distance_to_goal = sqrt((105 - gk_x)**2 + (34 - gk_y)**2)`.
3. `distance_to_shot = sqrt((anchor_x - gk_x)**2 + (anchor_y - gk_y)**2)`.
4. Stack into 4-column output.

NaN propagates through `sqrt(nan**2 + ...)` correctly.

### 4.4 Per-feature degradation policy (NaN semantics)

| Feature | Anchor | NaN cases | Zero cases |
|---|---|---|---|
| `pre_shot_gk_x` | GK's frame row | non-shot OR unlinked OR pre-engagement OR GK-absent-from-frame | (none — coordinate value) |
| `pre_shot_gk_y` | GK's frame row | same as above | (none) |
| `pre_shot_gk_distance_to_goal` | GK's frame row + goal center | same as above | (rare — GK exactly at goal center) |
| `pre_shot_gk_distance_to_shot` | GK's frame row + shot anchor | same as above | (rare — GK exactly at shot anchor) |

**Coordinate-convention contract:** identical to ADR-005 §5. `add_pre_shot_gk_position` requires `actions` and `frames` to use the same convention (recommend LTR-normalized via `spadl.play_left_to_right` + `tracking.play_left_to_right`). When called via `VAEP.compute_features(frames=...)`, both calls are run symmetrically before dispatch — typical user path gets normalization for free.

**Off-pitch values pass through unchanged.** Memory `feedback_lakehouse_consumer_not_source` — silly-kicks does not clamp provider-native coordinates. Bound checks in tests are broad ([-5, 110] for x; [-5, 73] for y) to acknowledge per-provider asymmetry (e.g., 15% Sportec ball-x off-pitch tail; SkillCorner broadcast geometry).

### 4.5 TF-11 — baselines backfill + per-row regression gate

#### 4.5.1 Slim parquet expected-output companion

```
tests/datasets/tracking/action_context_slim/
├── sportec_slim.parquet              # existing — input actions + frames (untouched)
├── metrica_slim.parquet              # existing (untouched)
├── skillcorner_slim.parquet          # existing (untouched)
├── sportec_expected.parquet          # NEW — expected add_action_context + add_pre_shot_gk_position output
├── metrica_expected.parquet          # NEW
├── skillcorner_expected.parquet      # NEW
└── pff_expected.parquet              # NEW — synthetic (from medium_halftime.parquet)
```

Existing input parquets are byte-stable across PR-S21 (preserves PR-S20 cross-provider tests). Expected-output parquets are TF-11's own regression target.

Each `*_expected.parquet` schema (~13 columns, < 5 KB each):

```
action_id                        int64        primary key for join
nearest_defender_distance        float64
actor_speed                      float64
receiver_zone_density            Int64
defenders_in_triangle_to_goal    Int64
pre_shot_gk_x                    float64      NEW per TF-1
pre_shot_gk_y                    float64
pre_shot_gk_distance_to_goal     float64
pre_shot_gk_distance_to_shot     float64
frame_id                         Int64
time_offset_seconds              float64
link_quality_score               float64
n_candidate_frames               int64
```

#### 4.5.2 Regenerator script (`scripts/regenerate_action_context_baselines.py`)

Deliberate-invocation only. Idempotent. Per provider:

1. Load committed `*_slim.parquet` (input actions + frames).
2. Run `add_pre_shot_gk_context(actions)` to populate `defending_gk_player_id` (events-only step).
3. Run `add_action_context(actions, frames)` (PR-S20 aggregator).
4. Run `add_pre_shot_gk_position(actions_with_gk_id, frames)` (PR-S21 aggregator).
5. Merge step-3 and step-4 output, project to expected schema.
6. Write `*_expected.parquet` (overwrites).
7. Compute `p25/p50/p75/p99` per feature; populate JSON null slots.

Run: `uv run python scripts/regenerate_action_context_baselines.py`.

Not part of CI. Output is reviewer-visible parquet diff via `git diff` parquet-aware tooling or a thin "dump-as-CSV-for-diff" companion (defer to plan).

#### 4.5.3 Per-row regression gate (load-bearing CI test)

`tests/tracking/test_action_context_expected_output.py` — NEW. Parametrized over the 4 providers:

```python
@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_add_action_context_per_row_regression(provider):
    actions, frames = _load_slim_inputs(provider)
    actions = add_pre_shot_gk_context(actions)            # events-only step
    actual_action_context = add_action_context(actions, frames)
    actual_gk = add_pre_shot_gk_position(actions, frames)
    actual = _project_to_expected_schema(actual_action_context, actual_gk)

    expected = pd.read_parquet(
        f"tests/datasets/tracking/action_context_slim/{provider}_expected.parquet"
    )
    pd.testing.assert_frame_equal(actual, expected, atol=1e-9, rtol=0, check_dtype=True)
```

Bit-exact, deterministic, no statistical tolerance.

#### 4.5.4 JSON baselines — role and gate

JSON null slots filled by the same regenerator. Role:
- **Documentation:** human-readable distribution summary committed alongside the data.
- **Schema-shape gate:** existing PR-S20 advisory-status assertion promoted to hard CI assert (4 features × 4 percentiles × 4 providers, no nulls, no NaN).
- **NOT a per-percentile drift assert.** Percentile-level drift is covered by the bit-exact per-row gate at §4.5.3.

`tests/tracking/test_empirical_action_context_baselines.py` — NEW (or extension of the existing PR-S20 file if one exists; verify in plan):

```python
def test_baselines_json_is_complete():
    """Every feature × provider × percentile populated; no nulls; no NaN."""

def test_baselines_json_matches_expected_parquet_distribution():
    """Sanity: percentiles in JSON computed from *_expected.parquet match within strict tolerance.
    Catches accidental hand-edits to JSON that drift from the data.
    Strict tolerance (atol=1e-6, rtol=0) — JSON-vs-parquet consistency, not feature-vs-baseline."""
```

#### 4.5.5 Coverage gap acknowledgment

Existing slim slices were sampled randomly within 30s windows; some providers may have 0 shots. Impact: GK-feature per-row regression gate verifies only the all-NaN path on those providers (still a useful regression check). Mitigation logged at §6.1 (R1).

### 4.6 Per-provider expected behavior summary

Per the empirical-probe baselines (PR-S19's `tests/datasets/tracking/empirical_probe_baselines.json` + PR-S20's slim-slice probe):

| Provider | Frame rate | Linkage rate (slim) | GK player_id stability | Off-pitch GK risk |
|---|---|---|---|---|
| PFF | 30 Hz | 1.0 | high (jersey-stable) | low |
| Sportec | 25 Hz | 1.0 | high | low (15% ball-x off-pitch tail per memory; GK rarely affected) |
| Metrica | 25 Hz | 1.0 | high (player_id stable; ~77% NaN ball-coords per probe — irrelevant to GK rows) | low |
| SkillCorner | 10 Hz | 1.0 | medium (broadcast tracking — occasional ID swaps) | medium (broadcast geometry — GK can land near goal-line edge) |

The per-provider asymmetries are absorbed by the bound-check ranges in the cross-provider parity test (§ 5.8).

## 5. Testing & validation

### 5.1 Test file deltas

```
tests/
├── tracking/
│   ├── test_kernels.py                              # extend: _pre_shot_gk_position analytical-truth tests
│   ├── test_features_standard.py                    # extend: 4 GK Series wrappers
│   ├── test_add_pre_shot_gk_position.py             # NEW — aggregator + provenance + nan_safe_enrichment auto-discovery
│   ├── test_action_context_expected_output.py       # NEW — per-row regression gate (TF-11 §4.5.3)
│   ├── test_empirical_action_context_baselines.py   # NEW — JSON shape + JSON-vs-parquet consistency (TF-11 §4.5.4)
│   ├── test_feature_framework.py                    # extend: ActionFrameContext.defending_gk_rows field test
│   ├── test_action_context_cross_provider.py        # extend: GK-feature bounds where shots exist
│   └── test_action_context_real_data_sweep.py       # extend: GK aggregator on full match data (e2e)
├── atomic/
│   ├── tracking/
│   │   ├── test_features_atomic.py                  # extend: 4 GK Series wrappers parametrized [standard, atomic]
│   │   └── test_add_atomic_pre_shot_gk_position.py  # NEW — atomic aggregator
│   └── ...
├── spadl/
│   └── test_add_pre_shot_gk_context.py              # extend: frames=None backcompat + frames=supplied
├── atomic/
│   └── test_atomic_add_pre_shot_gk_context.py       # extend: same as standard
├── vaep/
│   ├── test_compute_features_frames_kwarg.py        # extend: pre_shot_gk_default_xfns dispatch
│   └── test_hybrid_with_tracking.py                 # extend: AUC uplift augmented + GK xfns
└── (existing test_enrichment_nan_safety.py auto-discovers add_pre_shot_gk_position)
```

### 5.2 Tier-1 analytical kernel tests (`test_kernels.py` extension)

Per ADR-005, kernels are schema-agnostic and individually testable. ~6 new kernel tests:

1. Shot row, GK in linked frame at known coords → exact 4 outputs.
2. Non-shot row → all 4 NaN.
3. Shot row, `defending_gk_player_id` is NaN → all 4 NaN (pre-engagement case).
4. Shot row, GK player_id absent from linked frame → all 4 NaN (substitution case).
5. Shot row, action unlinked → all 4 NaN.
6. GK on edge of pitch (off-pitch x or y) → coords passed through as-is, distances computed correctly. Verifies no clamping (memory `feedback_lakehouse_consumer_not_source`).

Each fixture is ~3 frame rows + 1-2 actions, written inline.

### 5.3 Schema-wrapper parametrized parity (`test_features_standard.py` + `test_features_atomic.py`)

```python
@pytest.mark.parametrize("schema", ["standard", "atomic"])
def test_pre_shot_gk_distance_to_shot_anchor_correctness(schema):
    """Standard anchors on (start_x, start_y); atomic on (x, y). Both compute distance correctly."""
```

### 5.4 Aggregator + provenance tests (`test_add_pre_shot_gk_position.py`)

```python
def test_add_pre_shot_gk_position_emits_4_features_plus_4_provenance():
def test_add_pre_shot_gk_position_raises_on_missing_defending_gk_player_id_column():
def test_add_pre_shot_gk_position_idempotent_when_called_twice():
def test_add_pre_shot_gk_position_nan_safe_per_adr_003():
    # Auto-covered via @nan_safe_enrichment + tests/test_enrichment_nan_safety.py auto-discovery.
def test_add_pre_shot_gk_position_provenance_columns_match_link_actions_to_frames_pointers():
```

Atomic mirror in `tests/atomic/tracking/test_add_atomic_pre_shot_gk_position.py`.

### 5.5 Events-side wrapper tests (extending `test_add_pre_shot_gk_context.py`)

```python
def test_add_pre_shot_gk_context_frames_none_bit_identical_to_pr_s20():
    """No frames: output bit-equal to silly-kicks 2.8.0 — pinned via golden fixture."""

def test_add_pre_shot_gk_context_frames_supplied_emits_4_extra_columns_plus_provenance():

def test_add_pre_shot_gk_context_frames_supplied_no_module_import_cycle():
    """Importing silly_kicks.spadl.utils alone does NOT eagerly import silly_kicks.tracking.* —
    lazy import gate per ADR-005 §5."""

def test_add_pre_shot_gk_context_with_frames_handles_nan_defending_gk_player_id():
    """ADR-003: NaN defending_gk_player_id → NaN GK-position, no crash."""
```

Atomic mirror in `test_atomic_add_pre_shot_gk_context.py`.

### 5.6 VAEP integration tests

`test_compute_features_frames_kwarg.py` extension:
```python
def test_compute_features_dispatches_pre_shot_gk_default_xfns():
    """xfns=pre_shot_gk_default_xfns; frames supplied → 4 columns × nb_states emitted."""
```

`test_hybrid_with_tracking.py` extension:
```python
def test_hybrid_vaep_auc_uplift_with_gk_features():
    """xfns = hybrid_xfns_default + tracking_default_xfns + pre_shot_gk_default_xfns
    Assert AUC_with_gk >= AUC_without_gk + epsilon (epsilon=0.005 on synthetic)."""
```

The AUC uplift threshold is small because synthetic data has limited GK signal; the test validates wiring + non-degenerate signal, not magnitude.

### 5.7 TF-11 regression gate tests

Specified in §4.5.3-4.5.4. Bit-exact `assert_frame_equal` per provider; JSON shape + JSON-vs-parquet consistency.

### 5.8 Cross-provider parity (`test_action_context_cross_provider.py` extension)

Bounds-only assertions where shots exist:

```python
@pytest.mark.parametrize("provider", ["pff", "sportec", "metrica", "skillcorner"])
def test_pre_shot_gk_position_bounds(provider):
    """Where shots exist:
       - pre_shot_gk_distance_to_goal in [0, ~125] (pitch diagonal)
       - pre_shot_gk_distance_to_shot in [0, ~125]
       - pre_shot_gk_x in [-5, 110] (off-pitch tail allowed per provider asymmetry)
       - pre_shot_gk_y in [-5, 73]"""
```

Off-pitch tolerance acknowledges memory `reference_lakehouse_tracking_traps` (15% Sportec ball-x off-pitch; SkillCorner broadcast geometry).

### 5.9 Public-API Examples coverage

Every new public def gets an `Examples` section. `tests/test_public_api_examples.py` auto-discovers the new modules. `add_pre_shot_gk_context`'s extended docstring (with `frames=` kwarg) gets an updated Examples block showing both paths. Per memory `feedback_public_api_examples`.

### 5.10 e2e real-data sweep (`test_action_context_real_data_sweep.py` extension)

Already-existing parametrized e2e test gains GK-feature coverage on full match data. Same skip-on-missing-env-var behavior, with the loud-skip pattern from `feedback_no_silent_skips_on_required_testing`. Local paths come from runtime memory resolution, NOT from spec/code (see § 6.5).

### 5.11 Pre-commit gates

- `ruff` + `pyright` + `pytest -m "not e2e"` (Shift Left, memory).
- `tests/test_action_context_real_data_sweep.py` run locally with all four `*_TRACKING_DIR` env vars set.
- `/final-review` (mandatory per `feedback_final_review_gate`).
- One commit per branch (squash merge per `feedback_commit_policy`).

## 6. Risks, open items, execution-session expectations

### 6.1 Risks

**R1. Slim-slice shot coverage may be 0 for some providers.** Sportec/Metrica/SkillCorner slim slices were sampled randomly within 30s windows; some may contain no shot-class actions. Impact: GK-feature per-row regression gate verifies only the all-NaN-when-no-shots path on those providers (still a useful regression check).

Mitigation (Loop 0 of execution session): probe shot counts. If `< 1` for any provider, two paths:
- (a) **Default**: Augment the per-provider `_expected.parquet` with the existing all-NaN expected output (acceptable; analytical kernel correctness is verified in Tier-1 tests).
- (b) Re-sample that provider's slim slice to bias toward shot-containing windows. **Avoid (b) unless absolutely necessary** — cascades through `match_key_sampled` in JSON and breaks PR-S20 cross-provider linkage assumptions.

PFF synthetic `medium_halftime.parquet`: if 0 shots, supplement with a tiny `medium_halftime_with_shot.parquet` rather than re-generating the existing fixture.

**R2. ADR-005 `ActionFrameContext` field addition is a public-API change.** The dataclass is `@dataclasses.dataclass(frozen=True)`; adding a field is non-breaking for *attribute access* (existing accesses unchanged). Mitigation: all construction goes through `_resolve_action_frame_context` (the public builder); document in the dataclass docstring that direct construction is not part of the public API.

**R3. Pyright + `defending_gk_player_id` typing.** Column is `float64` (NaN-coded int per existing helper); when filtered against frame `player_id` (int or float per provider), pandas comparison surfaces a dtype-mismatch warning under strict typing. Mitigation: cast both sides to `float64` in the `_resolve_action_frame_context` extension (NaN-safe).

**R4. Pyright + lazy import via `TYPE_CHECKING`** may behave differently between local Python 3.14 and CI 3.10/3.11/3.12. Memory `feedback_python314_pandas_gotchas` flags this class of issue. Mitigation: pin pyright + pandas-stubs to CI versions before relying on local pyright (memory `feedback_ci_cross_version`).

**R5. Backward-compat-bit-identity is load-bearing.** `add_pre_shot_gk_context(actions)` (no frames) MUST produce identical output to silly-kicks 2.8.0. Mitigation: golden-fixture test (small actions DataFrame, expected output committed as parquet) pinned across pre-PR-S21 and post-PR-S21. New test reads the golden fixture and asserts.

**R6. `defending_gk_player_id` is not always populated when callers run only `add_action_context` (PR-S20).** A user appending `pre_shot_gk_default_xfns` to their VAEP xfns without first running `add_pre_shot_gk_context` on their actions gets a `ValueError` at first frame-aware xfn call. Mitigation: documented in `pre_shot_gk_default_xfns` docstring + `ValueError` with helpful message ("call `add_pre_shot_gk_context(actions)` first to populate `defending_gk_player_id`"). Surface loudly at boundary, ADR-005 §2 pattern.

### 6.2 Open items (resolve in execution session B Loop 0)

- **OI-1.** Probe slim-slice shot counts; decide R1 mitigation per provider.
- **OI-2.** Verify PFF `medium_halftime.parquet` has shot rows; if not, prepare augmented fixture.
- **OI-3 (resolved in spec).** `pre_shot_gk_default_xfns` is a SEPARATE list, not appended into `tracking_default_xfns`. Composability + GK shot-only signal would force `defending_gk_player_id` requirement on all PR-S20 callers if merged.

### 6.3 Out of scope (deferred to follow-up cycles)

- **GK angle features** → TF-12 (logged in this PR per the National Park bundle below).
- **Frame-based GK identification** → TF-13 (logged in this PR).
- **Defensive-line features** (line height, compactness, line break) → TF-14 (logged in this PR).
- **Pressure-on-keeper** → already covered by TF-2 (`pressure_on_actor` is generic-actor; the keeper is just one actor).
- **Lakehouse boundary adapter** → TF-10.

### 6.4 Versioning

- silly-kicks **2.9.0** (MINOR — additive feature surface; no breaking changes).
- Wheel + sdist publish via PyPI per `project_release_state`.
- Tag `v2.9.0` after merge SHA; user runs `git tag` + `git push origin v2.9.0`.

### 6.5 Execution-session expectations

- **One commit per branch** (memory `feedback_commit_policy`). Spec + plan + implementation + tests all in one squash commit.
- **`/final-review`** mandatory before commit (memory `feedback_final_review_gate`).
- **Empirical validation**: `pytest -m "not e2e"` green, then `pytest -m e2e` green on full real data. Memory `feedback_empirical_validation_before_ship` — synthetic-fixture green is necessary but not sufficient; sweep before commit.
- **Tracking-dir env vars** (per `feedback_no_silent_skips_on_required_testing`): `PFF_TRACKING_DIR`, `IDSSE_TRACKING_DIR`, `METRICA_TRACKING_DIR`, `SKILLCORNER_TRACKING_DIR`. **Local paths come from memory at runtime, NOT from spec/plan/code/CHANGELOG.** The execution session reads the local path from memory references (`reference_pff_data_local`, lakehouse tracking references), sets the env var in its shell, and runs the e2e sweep. Spec/plan/code reference only the env-var name.
- **Loud surface of skips**: any test that pytest-skips at e2e time gets surfaced inline before `/final-review`, not buried in summary. Remediate inline using runtime path resolution from memory.
- **Cross-version checks**: pin pyright + pandas-stubs to CI versions (Python 3.10/3.11/3.12) before relying on local 3.14 type-check (memory `feedback_ci_cross_version`).
- **No new branch creation in this session.** Branch `feat/pre-shot-gk-plus-baselines` created as Task 0 of execution session B's plan.

### 6.6 Sign-off criteria

- All Tier-1 + Tier-2 + Tier-3 tests green.
- e2e sweep green across all 4 providers.
- `/final-review` clean.
- Backward-compat golden-fixture test pins `add_pre_shot_gk_context(actions)` (no frames) bit-identical to 2.8.0.
- AUC uplift test passes (HybridVAEP with GK xfns ≥ baseline + ε on synthetic).
- Public-API Examples coverage 100%.
- NOTICE updated, TODO.md updated (TF-1 + TF-11 SHIPPED, TF-12 + TF-13 + TF-14 added), CHANGELOG entry added.

## 7. NOTICE + TODO.md updates

### 7.1 NOTICE — extend Anzer & Bauer entry (Fork 3)

Single-line description swap:

```diff
 - Anzer, G., & Bauer, P. (2021). "A goal scoring probability model for shots
   based on synchronized positional and event data in football and futsal."
   Frontiers in Sports and Active Living, 3, 624475.
-  (player_speed and distance-to-defender as xG features)
+  (player_speed, distance-to-defender, and defending-GK-position as xG features)
```

Cross-link from per-feature docstrings via `See NOTICE for full bibliographic citations.`

The four GK-feature docstrings each get a `References` section pointing at Anzer & Bauer (2021). Atomic-side docstrings reference the same. `add_pre_shot_gk_context`'s existing docstring gains a one-line addition to its References section pointing at Anzer & Bauer (2021) for the GK-position derivation when `frames=` is supplied; the current Butcher et al. (2025) reference for engagement-state stays.

### 7.2 TODO.md updates

1. **Mark TF-1 + TF-11 SHIPPED:** remove from On-Deck table.

2. **Active Cycle archival:** replace PR-S20 Active Cycle section with PR-S21:

```markdown
## Active Cycle

PR-S21 — TF-1 (`pre_shot_gk_position_*`) + TF-11 (baselines backfill)
(target silly-kicks 2.9.0).

Branch: `feat/pre-shot-gk-plus-baselines`. Spec + plan: [docs/superpowers/specs/2026-05-01-pre-shot-gk-plus-baselines-design.md](...), [docs/superpowers/plans/2026-05-01-pre-shot-gk-plus-baselines.md](...).

After ship, this section gets archived; PR-S21 is expected to ship within
the existing ADR-005 envelope — no new ADR.
```

3. **Header date update.**

4. **National Park bundle — add TF-12 / TF-13 / TF-14 to On-Deck table:**

```markdown
| TF-12 | `pre_shot_gk_angle_*` (signed angle from goal-line normal, off-line angular displacement, etc.) | Dunkin' | Anzer & Bauer 2021; PR-S21 deferral | Library ships positions + 2 distances in PR-S21; angle conventions deferred. Multiple competing definitions (relative to shot trajectory? to goal-line normal? signed vs unsigned?) — pick one canonical convention with downstream reviewer input before landing. ~30-50 LOC. |
| TF-13 | Frame-based defending-GK identification (fallback when events-based `defending_gk_player_id` is NaN) | Wicked | Bauer & Anzer 2021 (Section 3 carrier-ID heuristic, similar shape); Bekkers 2024 (DEFCON GK identification) | Heuristic: defender closest to own goal at the linked frame, possibly conditional on jersey/role data when supplied. Composes with PR-S21's strict events-only ID (callers opt into fallback). ~80-120 LOC + ADR if chosen heuristic is contentious. |
| TF-14 | Defensive-line features (line height, line compactness, line break detection) | Wicked | Power et al. 2017 (line break in OBSO); Spearman 2018; Anzer & Bauer 2021 | Per-frame defending team's outfield line geometry (median y of back-4, std dev, max gap). Could replace ad-hoc "defenders behind the ball" features in xG. ~150 LOC. |
```

### 7.3 CHANGELOG.md (execution session, not this session)

Spec doesn't pre-write the CHANGELOG entry — that's a Loop-N task in the implementation plan. Illustrative shape (NOT committed in this session):

```markdown
## 2.9.0 — 2026-05-XX (PR-S21)

### Added
- `silly_kicks.tracking.features.pre_shot_gk_x/_y/_distance_to_goal/_distance_to_shot` — 4 GK-position features at the linked frame.
- `silly_kicks.tracking.features.add_pre_shot_gk_position` — aggregator + 4 linkage-provenance columns. NaN-safe per ADR-003.
- `silly_kicks.atomic.tracking.features` mirrors of all of the above.
- `pre_shot_gk_default_xfns` — composable into HybridVAEP / AtomicVAEP via xfn list append.
- `silly_kicks.spadl.utils.add_pre_shot_gk_context(*, frames=None)` — additive optional kwarg; emits 4 GK-position columns + 4 provenance columns when frames supplied. Bit-identical when `frames=None` (backward-compat preserved). Atomic mirror.
- `tests/datasets/tracking/action_context_slim/{provider}_expected.parquet` — per-provider expected output for the per-row regression gate.
- `scripts/regenerate_action_context_baselines.py` — one-shot regenerator for the JSON baselines + `*_expected.parquet`.

### Changed
- `silly_kicks.tracking.feature_framework.ActionFrameContext` gains `defending_gk_rows: pd.DataFrame` field.
- `tests/datasets/tracking/empirical_action_context_baselines.json` — 64 null slots backfilled (4 percentiles × 4 features × 4 providers).
- NOTICE — Anzer & Bauer (2021) entry description expanded.
```

### 7.4 ADR status

No new ADR. PR-S21 ships entirely within ADR-005's 7 cross-cutting decisions. The `ActionFrameContext` extension is a non-breaking field addition that conforms to ADR-005 §3 (kernel-extraction pattern). The events-side helper extension uses §5 lazy-import pattern verbatim. ADR-005 itself stays untouched (optionally: add a sentence in "Consequences → Positive" mentioning PR-S21 as a concrete inheritance — defer to plan).

## References

- `docs/superpowers/specs/2026-04-30-action-context-pr1-design.md` — PR-S20 design (the feature precedent).
- `docs/superpowers/adrs/ADR-005-tracking-aware-features.md` — the 7 cross-cutting decisions PR-S21 inherits.
- `docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md` — the 9 invariants PR-S20 inherits and PR-S21 by extension.
- `docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md` — NaN-safety contract for enrichment helpers (`add_pre_shot_gk_position`, extended `add_pre_shot_gk_context`).
- ADR-001 — converter identifier conventions (cross-namespace consistency precedent).
- Anzer, G., & Bauer, P. (2021). "A goal scoring probability model for shots based on synchronized positional and event data in football and futsal." Frontiers in Sports and Active Living, 3, 624475. (Primary methodological reference for TF-1; existing NOTICE entry expanded.)
- Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). "Actions Speak Louder Than Goals." Proc. KDD '19. (Foundational VAEP.)
- See `NOTICE` for full bibliographic citations.
