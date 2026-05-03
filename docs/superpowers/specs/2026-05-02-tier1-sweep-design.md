# Tier 1 sweep — TF-6 + TF-8 + TF-9 + TF-12 — PR-S24 — silly-kicks 3.1.0

**Status:** Draft v3 (design — lakehouse-session feedback rounds 1+2 folded in 2026-05-02; pending user review)
**Target release:** silly-kicks 3.1.0 (additive surface; no breaking changes)
**Author:** Karsten S. Nielsen with Claude Opus 4.7 (1M)
**Date:** 2026-05-02
**Predecessor:** silly-kicks 3.0.1 — PR-S23 (Sportec + Metrica per-period direction-of-play correctness)
**Successors:** TF-2 (`pressure_on_actor` multi-flavor), TF-3 (`actor_distance_pre_window`), TF-5 (`infer_ball_carrier`), TF-7 (pitch control — own scoping cycle), TF-13/14/15-19 (GKDV research arc — Tier 3+)

**Triggers:**

- TODO.md On-Deck Tier 1 (post-2026-05-02 reorganization) — four Dunkin'-sized items flagged spec-complete: TF-6 (`sync_score`), TF-8 (smoothing), TF-9 (interpolation), TF-12 (`pre_shot_gk_angle_*`).
- User directive 2026-05-02: bundle the entire Tier 1 into one PR cycle since each row is "ready to ship."
- Best-practice / long-term scope override (`feedback_engineering_disciplines`): downstream Tier 3-6 features must be facilitated, not just the 4 items individually.
- Post-mortem on 3.0.0 / 3.0.1: both shipped with synthetic-only validation and were corrected for **geometric-correctness blind spots** (per-period direction-of-play). Any new geometric / signal-processing feature this PR ships must close that gap by construction.
- silly-kicks announced publicly on LinkedIn 2026-05-01 — visibility raises the testing bar.

---

## 1. Problem

Four Tier 1 On-Deck items are individually spec-complete but have never been bundled. Shipping them piecemeal risks:

1. **Re-litigating preprocessing conventions per feature.** TF-7 (pitch control), TF-15 (GK reachable area), TF-4 (off-ball runs), and the entire GKDV arc all depend on **smoothed positions and derived velocities**. Without a shared preprocessing pipeline (TF-8 + TF-9 + a config object), each downstream feature either re-implements smoothing or documents a "user must call X first" UX cliff. The cliff scales badly.

2. **Re-litigating QA gating per feature.** TF-6 (`sync_score`) is a foundational QA primitive. Downstream tracking-aware features (TF-13 frame-based GK ID, TF-15 GK reachable area, TF-7 pitch control) want to gate compute on link quality — `if sync_score < threshold: NaN`. Shipping it standalone unblocks every Tier 3-6 row.

3. **Re-litigating GK-geometry conventions per feature.** TF-12 completes the geometric set started by PR-S21 (`pre_shot_gk_position` 4 columns). Without angles, downstream xG / xGOT / GKDV consumers compute angles locally with inconsistent conventions, defeating the lakehouse-cross-pipeline-parity discipline.

4. **Re-litigating multi-flavor xfn conventions.** TF-2 will set the multi-flavor `method=` precedent for VAEP feature xfns (suffixed columns). TF-8/TF-9 are not VAEP feature xfns but are multi-method utilities (SG vs EMA; linear vs cubic). Without an explicit asymmetry rule, future spec sessions will re-debate per item.

5. **Re-litigating empirical-baseline strategy per feature.** Picking SG window, EMA alpha, max_gap_seconds, and "high-quality" threshold without measuring real per-provider data is exactly the synthetic-only validation pattern that produced 3.0.0 / 3.0.1.

This PR (PR-S24) lands all four Tier 1 items under one cohesive design that addresses each re-litigation risk explicitly.

## 2. Goals

1. **Four feature-deliverables**, all additive (no API breaks):
   - **TF-6 — `sync_score`**: per-action tracking↔events sync-quality, three aggregations (`min` / `mean` / `high_quality_frac`) returned together. Available as pure helper, `add_sync_score(...)` mutator, and `LinkReport.sync_scores()` method.
   - **TF-8 — smoothing primitives**: `smooth_frames(frames, *, method="savgol"|"ema", ...)` and `derive_velocities(frames, ...)` in a new `silly_kicks.tracking.preprocess` module. Single canonical output columns (`vx`, `vy`, `speed`); method recorded as metadata.
   - **TF-9 — interpolation / gap-filling**: `interpolate_frames(frames, *, method="linear"|"cubic", max_gap_seconds=...)` in the same `preprocess` module.
   - **TF-12 — GK angle features**: `add_pre_shot_gk_angle(actions, *, frames=...)` aggregator emitting 2 columns (`pre_shot_gk_angle_to_shot_trajectory`, `pre_shot_gk_angle_off_goal_line`, both signed). Standalone aggregator; unsigned variants are caller-side `abs()`.

2. **Shared preprocessing config object** (`PreprocessConfig` dataclass). Single source of truth for smoothing / interpolation / velocity-derivation parameters; per-provider defaults table (`_PROVIDER_DEFAULTS`) populated from probe baselines. Tier 3-6 specs can reference `PreprocessConfig.default()` instead of redefining preprocessing per feature.

3. **Optional converter wiring with auto-promotion**. Each tracking converter gains one new kwarg `preprocess: PreprocessConfig | None = None`. Default `None` ⇒ zero behaviour change (backcompat with 3.0.x). When set, the converter delegates to the `preprocess` utilities after frame extraction. **Auto-promotion (flag-based, not value-equality)**: if `preprocess.is_default()` returns True, the converter auto-promotes to `PreprocessConfig.for_provider(self.provider_name)`. The `is_default()` check reads a private `_is_universal_default: bool` flag that is set ONLY by the `PreprocessConfig.default()` factory — a hand-constructed `PreprocessConfig(sg_window_seconds=0.4, ...)` with values coincidentally matching universal defaults returns `is_default() == False` and is NOT auto-promoted (consumer's explicit construction intent preserved; no silent footgun). Consumers who genuinely want universal defaults from a provider-aware caller pass `PreprocessConfig.default(force_universal=True)`. Documented in each converter's docstring.

4. **Umbrella facade for GK geometry, with explicit consumer-contract handshake**. Existing `silly_kicks.spadl.utils.add_pre_shot_gk_context` (events-side facade, lazy-imports the GK aggregator) extends to lazy-import **both** position and angle aggregators when `frames` is supplied — emits 6 columns total (4 from PR-S21 + 2 new). Existing `add_pre_shot_gk_position` is **unchanged** (no surprise schema extension).

**Consumer-contract handshake.** The 4→6 column expansion of `add_pre_shot_gk_context(actions, *, frames=...)` is nominally additive but BREAKS any consumer that asserts on column-set equality (notably lakehouse `src/tests/test_silly_kicks_boundary.py`). For 3.1.0 the handshake is:

- This spec is the pre-merge source of truth (lakehouse can read it now and pre-bump `expected_columns` in their boundary test in their own `silly-kicks>=3.1.0,<4` pin-bump PR).
- CHANGELOG entry under `### Added` enumerates the two new columns by exact name (`pre_shot_gk_angle_to_shot_trajectory`, `pre_shot_gk_angle_off_goal_line`) so the boundary-test pin can be ctrl-F'd from the diff.
- The umbrella facade behaviour when `frames=None` is **bit-identical** to silly-kicks 2.9.0 — emits exactly 4 columns. Lakehouse boundary tests that exercise the `frames=None` path need no change.
- No deprecation; existing 4-column callers via `add_pre_shot_gk_position` keep their exact surface.

5. **Multi-flavor convention asymmetry, codified.**
   - **VAEP feature xfns** (TF-2, future shot-features): suffixed column names (`__defcon`, `__ema_decay`, etc.) per the TF-2 precedent. Reason: parallel xfn registration would silent-overwrite same-named columns inside `VAEP.compute_features`.
   - **Preprocessing utilities** (TF-8, TF-9): single canonical column names (`vx`, `vy`, `speed`, `x_smoothed`, `y_smoothed`). Method recorded as a per-row `_preprocessed_with` provenance column (load-bearing — survives merge/concat/applyInPandas; cheap in dictionary-encoded Parquet) PLUS `frames.attrs["preprocess"]` as a convenience mirror for in-process introspection. Reason: downstream features depend on schema stability; method comparison is a separate research workflow (run twice, diff). Original raw `x`/`y` are preserved unchanged (Hyrum's Law protection — additive `_smoothed` columns rather than in-place mutation).
   - **QA helpers** (TF-6): single function with `aggregation=` parameter when aggregations are alternative views of the same metric; multi-column emit when aggregations answer **genuinely different questions** (TF-6 case — three views for three downstream consumers).
   - This asymmetry is documented in the new `tracking.preprocess` module docstring AND queued as an ADR-005 amendment to land alongside TF-2.

6. **Per-provider empirical-baseline strategy.**
   - Probe scripts in `scripts/probe_preprocess_baseline.py` measure: sampling rate (Hz), raw-position noise floor, velocity-outlier rate, gap-rate distribution, gap-length distribution, `link_quality_score` distribution, GK-angle distribution (sanity).
   - Output committed as `tests/fixtures/baselines/preprocess_baseline.json`.
   - `_PROVIDER_DEFAULTS` in code traces every value back to the JSON via comments; integrity test (`tests/test_preprocess_baseline_integrity.py`) asserts code defaults match JSON within tolerance.
   - Probes are re-runnable post-merge; parameter re-tuning goes through bump-baseline minor-version PR (deliberate, not silent).

7. **TDD-first**, RED-before-GREEN. Every helper opens with analytic-invariant tests before any implementation. Numeric features get registered `tests/invariants/` entries.

8. **Closes the 3.0.0 / 3.0.1 blind-spot pattern.** Per-period direction-of-play symmetry invariant added for TF-12 angles. Fixture-density gate ensures every invariant is physically exercised (no vacuous skip-passing).

9. **Empirical sweep against full real datasets** (PFF FC WC2022 + Sportec local + Metrica lakehouse-derived) BEFORE the single commit. Probe-output log captured in PR description.

10. **NOTICE attribution discipline**: extend existing entries (Anzer & Bauer 2021 for TF-12) and add new entries (Savitzky & Golay 1964 for TF-8) per ADR-005.

## 3. Non-goals

- **No multi-flavor TF-2 (`pressure_on_actor`) implementation.** TF-2 is Tier 2; the multi-flavor convention asymmetry is *codified* here, but TF-2 itself ships in its own PR.
- **No TF-7 pitch control.** TF-7 is the gate for the entire GKDV arc and warrants its own scoping cycle.
- **No TF-3 (`actor_distance_pre_window`).** Tier 2; uses `slice_around_event` already; standalone PR.
- **No frame-based defending-GK identification (TF-13).** Tier 3 heuristic; PR-S21's events-only `defending_gk_player_id` remains the GK-resolution path. When NaN, TF-12 angles emit NaN (honest, no fabrication — same convention as PR-S21).
- **No new ADR.** Multi-flavor convention asymmetry is queued as an **ADR-005 amendment** to land with TF-2 (the first VAEP-xfn case that actually exercises the suffixed-column rule). PR-S24 documents the rule in `tracking.preprocess` module docstring as the operational source.
- **No lakehouse boundary adapter.** Lakehouse repo is consumer; PR-S20 §11 carries that contract.
- **No new tracking-provider adapters.** Inherits PR-S19's 4-provider coverage.
- **No streaming / chunked processing.** ADR-004 invariant 1.
- **No CHANGELOG handwriting in this session.** Implementation plan (writing-plans) has CHANGELOG as a final loop task.
- **No converter-signature breaking changes.** `preprocess` kwarg is keyword-only, defaults to `None`, zero impact when unused.

## 4. Design

### 4.1 Architecture & module layout

```
silly_kicks/tracking/
├── preprocess/                    # NEW MODULE
│   ├── __init__.py                # public surface: smooth_frames, interpolate_frames,
│   │                              # derive_velocities, PreprocessConfig
│   ├── _config.py                 # PreprocessConfig dataclass + _PROVIDER_DEFAULTS
│   ├── _smoothing.py              # smooth_frames(method="savgol"|"ema")
│   ├── _interpolation.py          # interpolate_frames(method="linear"|"cubic")
│   └── _velocity.py               # derive_velocities (SG-derivative canonical)
├── linkage.py                     # extend: sync_score(links, *, aggregation=...) helper
│                                  #         add_sync_score(actions, links, ...) mutator
│                                  #         LinkReport.sync_scores() method
├── features.py                    # extend: 2 angle Series helpers + add_pre_shot_gk_angle
│                                  #         + pre_shot_gk_angle_default_xfns
│                                  #         + pre_shot_gk_full_default_xfns (union list)
├── _kernels.py                    # extend: _pre_shot_gk_angle kernel
├── sportec.py                     # extend: optional preprocess kwarg
├── pff.py                         # extend: optional preprocess kwarg
├── kloppy.py                      # extend: optional preprocess kwarg
└── (existing schema / _direction / utils unchanged)

silly_kicks/atomic/tracking/
├── features.py                    # mirror: 2 angle helpers + aggregator + xfn lists
└── (preprocess / linkage / sync_score consumed unchanged from non-atomic)

silly_kicks/spadl/utils.py         # extend: add_pre_shot_gk_context lazy-imports angle aggregator
                                   # when frames is supplied; emits 6 columns

scripts/
└── probe_preprocess_baseline.py   # NEW — generates preprocess_baseline.json

tests/fixtures/baselines/
└── preprocess_baseline.json       # NEW — per-provider measured stats

tests/
├── test_preprocess_baseline_integrity.py    # NEW — JSON ↔ _PROVIDER_DEFAULTS consistency
├── test_sync_score.py                       # NEW
├── test_smooth_frames.py                    # NEW
├── test_interpolate_frames.py               # NEW
├── test_derive_velocities.py                # NEW
├── test_preprocess_config.py                # NEW
├── test_pre_shot_gk_angle.py                # NEW
├── test_add_pre_shot_gk_context_extended.py # NEW (umbrella facade emits 6 cols)
├── atomic/
│   └── test_sync_score_atomic.py            # NEW — parity smoke test (atomic actions
│                                            #       consume add_sync_score unchanged)
└── invariants/
    ├── test_invariant_sync_score_bounds.py            # NEW
    ├── test_invariant_smooth_frames_idempotence.py    # NEW
    ├── test_invariant_smooth_frames_constant_signal.py # NEW
    ├── test_invariant_interpolate_passes_through.py   # NEW
    ├── test_invariant_velocity_physical_plausibility.py # NEW
    ├── test_invariant_gk_angle_bounds.py              # NEW
    └── test_invariant_gk_angle_per_period_dop_symmetry.py  # NEW (CLOSES 3.0.0/3.0.1 GAP)
```

### 4.2 TF-6 — `sync_score` API surface

**Pure helper (primitive):**

```python
def sync_score(
    links: pd.DataFrame,                # per-action linkage pointers (from link_actions_to_frames)
    *,
    high_quality_threshold: float = ...,  # data-derived from preprocess_baseline.json
) -> pd.DataFrame:
    """Per-action sync-quality scores (3 columns).

    Returns a DataFrame indexed by action_id with columns:
      sync_score_min — float in [0, 1], min(link_quality_score) per action.
      sync_score_mean — float in [0, 1], mean(link_quality_score) per action.
      sync_score_high_quality_frac — float in [0, 1], fraction of links with
        link_quality_score >= high_quality_threshold.
    """
```

**Convenience mutator** (mirrors `add_*` family):

```python
def add_sync_score(
    actions: pd.DataFrame,
    links: pd.DataFrame,
    *,
    high_quality_threshold: float = ...,
) -> pd.DataFrame:
    """Returns actions with three sync_score_* columns merged on action_id."""
```

**LinkReport method** (locality):

```python
class LinkReport:
    ...
    def sync_scores(self, *, high_quality_threshold: float = ...) -> pd.DataFrame:
        """Per-action sync_score DataFrame for this link batch."""
```

All three are 1-line wrappers around the same kernel. Default `high_quality_threshold` is read from `_PROVIDER_DEFAULTS[provider].link_quality_high_threshold` derived from probe baselines (per provider, not universal).

### 4.3 TF-8 — smoothing primitives

```python
@dataclass(frozen=True)
class PreprocessConfig:
    smoothing_method: Literal["savgol", "ema", None] = "savgol"
    sg_window_seconds: float = 0.4
    sg_poly_order: int = 3
    ema_alpha: float = 0.3
    interpolation_method: Literal["linear", "cubic", None] = "linear"
    max_gap_seconds: float = 0.5
    derive_velocity: bool = True
    # Provenance flag — set by default() factory only. Never compared by value.
    # Excluded from __eq__ / __hash__ so two configs with the same field values
    # are still equal regardless of which factory built them.
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> "PreprocessConfig":
        """Universal-safe defaults. Per-provider tuning via for_provider().

        force_universal=True is an escape hatch for the rare consumer that
        passes default() to a provider-aware caller AND genuinely wants
        universal-safe values (debugging cross-provider comparisons under
        fixed config). When True, sets _is_universal_default=False so the
        provider-aware caller does NOT auto-promote.
        """
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> "PreprocessConfig":
        """Provider-tuned defaults from preprocess_baseline.json.
        _is_universal_default is False (the dataclass default).
        """

    def is_default(self) -> bool:
        """True iff this config came from default() without force_universal=True.

        FLAG-BASED, not value-equality: a hand-constructed
        PreprocessConfig(sg_window_seconds=0.4, ...) with values that happen
        to coincide with universal defaults returns False — the consumer's
        explicit construction intent is preserved, no silent auto-promotion.
        """
        return self._is_universal_default

def smooth_frames(
    frames: pd.DataFrame,
    *,
    config: PreprocessConfig | None = None,
    method: Literal["savgol", "ema"] | None = None,
) -> pd.DataFrame:
    """Smooth player/ball position columns.

    Output schema extension (canonical column names — single-flavor; ADDITIVE,
    not in-place):
      x, y — UNCHANGED (raw values preserved). Hyrum's Law protection: any
        consumer reading frames["x"] post-smoothing gets the original raw
        value, never silently mutated.
      x_smoothed, y_smoothed — NEW columns; float64; smoothed values.
        Consumers opt in by reading the _smoothed columns explicitly.
      _preprocessed_with — str column; one repeated value per smoothed group;
        records method + config-hash for downstream introspection. Cheap in
        dictionary-encoded Parquet (~bytes per file regardless of row count).

    Idempotent: smoothing twice with the same config returns equal output
    (already-smoothed frames detected via _preprocessed_with column).
    """

def derive_velocities(
    frames: pd.DataFrame,
    *,
    config: PreprocessConfig | None = None,
) -> pd.DataFrame:
    """Adds vx, vy, speed columns derived via SG-derivative on smoothed positions.

    Schema extension (single-flavor):
      vx, vy — float64, m/s.
      speed — float64, m/s. = sqrt(vx**2 + vy**2).

    Requires smoothed positions. Detection signal: if "_preprocessed_with"
    is NOT in frames.columns (the load-bearing provenance column added by
    smooth_frames), derive_velocities applies smoothing inline per the
    supplied config (transparent to caller). The detection deliberately
    consults the per-row column rather than frames.attrs["preprocess"]
    because attrs does not survive merge/concat/applyInPandas (per
    feedback_pandas_attrs_dont_propagate). Always emits velocities when
    called directly — config.derive_velocity only controls auto-derivation
    by smooth_frames in the config-driven path.
    """
```

**Method recorded as a per-row provenance column, NOT just `frames.attrs`.** Reason: `pandas.DataFrame.attrs` does not propagate through `merge` / `concat` / `groupby` / `applyInPandas` / most non-trivial operations — lakehouse pipelines do all of these between converter output and feature compute, so `attrs` would silently drop and `derive_velocities` would fall back to default config (or fail). The per-row `_preprocessed_with` column survives every pandas operation that preserves row identity, and in dictionary-encoded Parquet the storage cost of one repeated string across N rows is essentially free (a few bytes per file regardless of row count).

`frames.attrs["preprocess"]` is ALSO populated as a convenience mirror (for in-process introspection where attrs has survived) but is NEVER the load-bearing source of truth. The `_preprocessed_with` column is.

Downstream features detect-and-fail-loudly on absence of `vx`/`vy`/`speed` / `x_smoothed` / `y_smoothed` columns and consult `_preprocessed_with` for config introspection. Per `feedback_vaep_feature_column_names_introspection`: silent-NaN convention only inside VAEP-introspection path; loud-raise everywhere else.

### 4.4 TF-9 — interpolation / gap-filling

```python
def interpolate_frames(
    frames: pd.DataFrame,
    *,
    config: PreprocessConfig | None = None,
    method: Literal["linear", "cubic"] | None = None,
) -> pd.DataFrame:
    """Fill NaN positional gaps up to max_gap_seconds.

    Output: same schema as input; NaN positions filled where gap-length
    <= max_gap_seconds. Gaps longer than max_gap_seconds remain NaN
    (no fabrication of long-occlusion data).

    Idempotent on already-filled frames.
    """
```

`max_gap_seconds` defaults from `_PROVIDER_DEFAULTS[provider].max_gap_seconds`. Per-provider because typical play-interruption duration differs; Metrica's ~77% NaN ball-coord rate (per `reference_lakehouse_tracking_traps`) is structurally different from PFF FC's gap shape.

**Metrica caveat (explicit consumer expectation).** Metrica's ~77% ball-coord NaN rate is *structural* (sampling pipeline), not occlusion. Most Metrica gaps far exceed any reasonable `max_gap_seconds` setting (0.48s would fill <5% of them per probe-baseline projection). Post-`interpolate_frames` Metrica output will remain heavily NaN on ball coords. Downstream features that consume ball position will inherit silent-NaN per the VAEP-introspection convention. **Specifically:** TF-7 pitch control directly consumes ball position; TF-15 GK reachable area transitively inherits via TF-7's threat-weighting. **Specifically NOT affected:** TF-12 angle features (shot anchor is `action.start_x/start_y` events-side, GK position from linked frame's player coords ~2-3% NaN, goal-mouth centre is fixed (105, 34) — no ball-coord dependency). Player-coord NaN propagation (~2-3% on Metrica) is a much milder per-frame concern shared by all features that read player positions and is NOT specific to Metrica's ball-coord structural rate. The probe baseline JSON includes `post_interpolation_nan_rate_*` per provider per column to make all of this auditable; spec §5.1 pre-merge gate asserts the post-interpolation NaN rate matches the JSON within tolerance.

### 4.5 TF-12 — GK angle features

```python
def add_pre_shot_gk_angle(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame,                  # required — angle has no events-only fallback
) -> pd.DataFrame:
    """Add 2 GK-angle columns at the linked frame for each shot action.

    Output columns (both signed, both float64, both in radians):

    pre_shot_gk_angle_to_shot_trajectory
        Signed angle between (a) the goal-centre-to-shot-anchor vector and
        (b) the GK-position-to-shot-anchor vector, measured in the LTR-normalized
        attacking direction. Zero ⇒ GK is on the shot trajectory line.
        Positive ⇒ GK is to the +y side of the shot line; negative ⇒ -y side.
        Use case: shot-stopping / xGOT-adjacent — measures GK alignment with
        the shooter→goal line.

    pre_shot_gk_angle_off_goal_line
        Signed angle between (a) the goal-line normal vector (the unit vector
        perpendicular to the defending goal line, pointing into the field of
        play) and (b) the vector from the goal-mouth centre to the GK position.
        Equivalently: arctan2(gk_y - 34, 105 - gk_x) in LTR-normalized
        coordinates with goal-mouth centre at (105, 34). Zero ⇒ GK is directly
        in front of goal-mouth centre on the goal-line normal. Positive ⇒ GK
        is offset to the +y (upper-sideline) side; negative ⇒ -y side.
        Use case: positioning / sweeping — measures GK lateral displacement
        from the goal-mouth-centre line of approach.

    NaN when defending_gk_player_id is NaN OR when the linked frame has no GK row.
    Mirrors PR-S21 NaN-honesty convention.

    Unsigned variants: caller-side abs(). Not shipped unless a concrete consumer asks.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> # See tests/test_pre_shot_gk_angle.py for runnable example.
    """
```

**Standalone aggregator** (per Q2 resolution) — does NOT extend `add_pre_shot_gk_position` (which keeps its 4 columns unchanged).

**Umbrella facade extension** — `spadl.utils.add_pre_shot_gk_context(actions, *, frames=None, ...)` lazy-imports BOTH `add_pre_shot_gk_position` and `add_pre_shot_gk_angle` when `frames` is supplied. Emits all 6 columns total. Backcompat: when `frames=None`, behaviour is bit-identical to silly-kicks 2.9.0 (PR-S21).

**VAEP integration:** ship `pre_shot_gk_angle_default_xfns` as a separate frame-aware xfn list, plus a union list `pre_shot_gk_full_default_xfns` that combines `pre_shot_gk_default_xfns` (PR-S21 positions/distances) + `pre_shot_gk_angle_default_xfns`. Models opt in at whatever granularity they want.

### 4.6 Per-provider defaults & probe artifacts

`tests/fixtures/baselines/preprocess_baseline.json` schema:

```json
{
  "_provenance": {
    "generated_by": "scripts/probe_preprocess_baseline.py",
    "generated_at": "2026-05-XX",  // filled at probe-run time
    "silly_kicks_version": "3.1.0-dev",
    "providers_probed": ["sportec", "pff", "metrica", "skillcorner"]
  },
  "sportec": {
    "sampling_rate_hz": 25.0,
    "raw_position_noise_floor_m": 0.04,
    "velocity_outlier_rate_at_max_12mps": 0.0008,
    "gap_rate_player_pct": 2.1,
    "gap_rate_ball_pct": 14.7,
    "gap_length_p50_frames": 1,
    "gap_length_p99_frames": 12,
    "post_interpolation_nan_rate_player_pct": 0.3,
    "post_interpolation_nan_rate_ball_pct": 1.2,
    "link_quality_score_p50": 0.92,
    "link_quality_score_p10": 0.71,
    "link_quality_high_threshold": 0.85,
    "gk_angle_to_shot_trajectory_p50_rad": 0.04,
    "gk_angle_off_goal_line_p50_rad": 0.06,
    "_derived_defaults": {
      "sg_window_seconds": 0.4,
      "sg_poly_order": 3,
      "ema_alpha": 0.3,
      "max_gap_seconds": 0.48
    }
  },
  "pff": { "...": "..." },
  "metrica": { "...": "..." },
  "skillcorner": { "...": "..." }
}
```

`_PROVIDER_DEFAULTS` in `tracking/preprocess/_config.py`:

```python
# Generated from tests/fixtures/baselines/preprocess_baseline.json.
# Re-run: uv run python scripts/probe_preprocess_baseline.py --provider <name>
_PROVIDER_DEFAULTS: dict[str, PreprocessConfig] = {
    "sportec": PreprocessConfig(
        sg_window_seconds=0.4,    # covers 10 frames @ 25 Hz; > 4σ noise floor
        sg_poly_order=3,
        ema_alpha=0.3,            # half-life ~2 frames
        max_gap_seconds=0.48,     # P99 gap = 12 frames @ 25 Hz
        ...
    ),
    "pff": ...,
    ...
}
```

Every numeric value carries a comment tracing it to the measured stat in the JSON. `tests/test_preprocess_baseline_integrity.py` asserts the trace is correct (no silent hand-edits).

### 4.7 Multi-flavor xfn convention asymmetry

Documented in `silly_kicks/tracking/preprocess/__init__.py` module docstring:

> **Column-naming convention.** Preprocessing utilities emit single canonical column names (`vx`, `vy`, `speed`) regardless of the smoothing/interpolation method chosen. The method is recorded as DataFrame metadata, not in column names.
>
> This deliberately diverges from the convention used by VAEP feature xfns (e.g., `pressure_on_actor__defcon`, `pressure_on_actor__andrienko_cone` per TF-2), where suffixed names are required because parallel xfn registration would silent-overwrite same-named columns inside `VAEP.compute_features`.
>
> Preprocessing has no equivalent constraint: downstream features (TF-7 pitch control, TF-15 GK reachable area, etc.) depend on schema stability and consume single canonical inputs. Method comparison is a separate research workflow — call preprocess twice with different configs into different DataFrames and diff.
>
> ADR-005 amendment formalising this asymmetry will land with TF-2. PR-S24 documents the rule operationally.

## 5. Testing strategy

The 3.0.0 / 3.0.1 post-mortem dictates explicit per-pillar coverage. Matrix below; every cell is a test that must be written **before** the corresponding implementation (TDD-RED).

| Pillar | TF-6 sync_score | TF-8 smoothing | TF-9 interpolation | TF-12 GK angle |
|--------|-----------------|----------------|---------------------|----------------|
| **Analytic invariants (TDD-RED)** | bounds [0,1]; min ≤ mean; high_quality_frac ∈ [0,1] | constant signal returns constant; polynomial deg ≤ poly_order returns unchanged; idempotence | passes through observed points; NaN beyond max_gap_seconds; idempotence | bounds [-π, π] / [-π/2, π/2]; angle of GK-on-shot-line = 0; sign flips when GK crosses line |
| **Per-period DOP-symmetry (closes 3.0.0/3.0.1)** | n/a | n/a | n/a | **REQUIRED** — same shot mirrored across periods produces correctly sign-flipped angles |
| **Fixture-density gate** | ≥1 link per provider in slim fixtures | ≥1 dropped-frame per provider for SG to exercise | ≥1 gap per provider exceeding 1 frame | ≥1 shot+keeper_save per provider per period (extends `feedback_synthesizer_shot_plus_keeper_save_pattern`) |
| **Lakehouse-derived CI fixtures** | Sportec + Metrica slim slices | Sportec + Metrica | Sportec + Metrica | Sportec + Metrica |
| **Empirical sweep (full real datasets, pre-commit)** | link_quality_score distribution per provider | smoothed-velocity max < 12 m/s for 99.9% of frames per provider | gap-fill rate matches per-provider baseline | angle distributions cluster near 0 with fat tail |
| **Idempotence** | sync_score(sync_score(...)) ≡ sync_score(...) | smooth(smooth(x)) ≈ smooth(x) within tol | interpolate(interpolate(x)) ≡ interpolate(x) | n/a |
| **Schema/dtype** | int64 action_id; float64 scores | float64 vx/vy/speed/x_smoothed/y_smoothed; raw x/y preserved unchanged; str `_preprocessed_with` column | preserves dtypes; raw NaN preserved | float64 angles in radians |
| **NaN propagation** | NaN action_id → NaN sync_score | NaN positions skipped by SG | NaN beyond max_gap stays NaN | NaN gk_player_id → NaN angles |
| **`SILLY_KICKS_ASSERT_INVARIANTS=1` registration** | wired | wired | wired | **wired (per-period DOP)** |
| **Public-API Examples** | required (3 entries) | required (2 entries) | required (1 entry) | required (1 entry) |
| **Atomic-SPADL parity** | atomic actions consume `add_sync_score` unchanged (operates on `action_id`, not action content); no separate atomic mirror needed; one parity smoke test in `tests/atomic/test_sync_score_atomic.py` confirms | n/a (preprocess universal — frames have no events/atomic distinction) | n/a | mirrored in `atomic.tracking.features` |

### 5.1 Pre-merge gates specific to PR-S24

1. **Probe scripts runnable** — `uv run python scripts/probe_preprocess_baseline.py --provider <name>` produces deterministic JSON output.
2. **`tests/test_preprocess_baseline_integrity.py`** passes — `_PROVIDER_DEFAULTS` numeric values match `preprocess_baseline.json` within explicit tolerances (revised 2026-05-02 per lakehouse review item #10): **exact match** for integer fields (`sg_poly_order`, `gap_length_p50_frames`, `gap_length_p99_frames`); **`math.isclose(..., rel_tol=1e-6, abs_tol=0.0)`** for float fields (`sg_window_seconds`, `ema_alpha`, `max_gap_seconds`, `link_quality_high_threshold`, `raw_position_noise_floor_m`, `velocity_outlier_rate_at_max_12mps`, all `*_pct` fields, all `*_p<NN>` percentile fields). Catches silent hand-edits of either artifact.
3. **Empirical sweep log committed** alongside the baseline JSON at `tests/fixtures/baselines/preprocess_sweep_log.json` (revised 2026-05-02 per lakehouse review item #9). Aggregate per-provider distribution stats only (no row-level data, no license-restricted real-data dumps): TF-12 angle p10/p50/p90, smoothed-velocity p99, gap-fill rates, post-interpolation NaN rates. Re-runnable via `scripts/probe_preprocess_baseline.py --emit-sweep-log`. Durable empirical basis for future debugging / parameter re-tuning. PR description still pastes the diff for human review.
4. **Per-period DOP-symmetry invariant** for TF-12 explicitly tested AND wired into `SILLY_KICKS_ASSERT_INVARIANTS=1` CI gate.
5. **Fixture-density audit** — `tests/test_invariant_density_audit.py` (extends existing pattern) asserts every invariant test reports ≥1 row tested per provider per period.
6. **Standard pre-PR gates** — `ruff format --check`, `ruff check`, `pyright`, full pytest, then `/final-review` (mandatory per `feedback_final_review_gate`).

## 6. ADR implications

- **No new ADR.** All four items fit within ADR-004 (tracking primitives) and ADR-005 (frame-aware xfn / composition / NOTICE attribution).
- **ADR-005 amendment queued** for TF-2 PR — formalises the multi-flavor convention asymmetry (suffixed columns for VAEP xfns; canonical-single for preprocessing utilities). PR-S24 documents the rule operationally; TF-2 PR codifies it as ADR text.

## 7. Versioning

- silly-kicks 3.1.0 — additive surface (new `tracking.preprocess` module; new `add_sync_score`, `add_pre_shot_gk_angle`; extended `add_pre_shot_gk_context`; new optional `preprocess` kwarg on tracking converters).
- No breaking changes; no migration notes required.
- CHANGELOG entry handwritten in implementation plan's final loop task.

## 8. Open implementation choices (defer to writing-plans)

- Number of loops in implementation plan (estimate: 5-7 loops — probe scripts → preprocess module → sync_score → GK angle → umbrella facade extension → VAEP integration → CHANGELOG/NOTICE).
- Exact ordering of TDD-RED test files vs. implementation modules.
- Whether to bundle the probe-script execution into a single `make probe` target or keep per-provider invocations.
- ~~Whether `_PROVIDER_DEFAULTS` for SkillCorner ships in PR-S24~~ — **resolved 2026-05-02 (lakehouse review)**: SkillCorner ships in PR-S24. Lakehouse A-League runs through SkillCorner via kloppy; deferring would force lakehouse to fall back to `default()` which is a poor fit given per-provider gap-shape variation. Conservative defaults backed by the PR-S19 probe baseline (already covered all 4 providers including SkillCorner in the e2e sweep). Per `feedback_no_silent_skips_on_required_testing`.

## 9. References

**TF-6 sync_score:** ADR-004 §4 (linkage primitives); novel utility, no academic citation.

**TF-8 smoothing:**
- Savitzky, A., & Golay, M. J. E. (1964). "Smoothing and Differentiation of Data by Simplified Least Squares Procedures." *Analytical Chemistry*, 36(8), 1627-1639.

**TF-9 interpolation:** Standard numerical-methods (cubic spline / linear); no domain-specific citation.

**TF-12 GK angle:**
- Anzer, G., & Bauer, P. (2021). "Expected Passes." *Frontiers in Sports and Active Living*, 3, 624475.

NOTICE entries extended: existing Anzer & Bauer 2021 bullet expanded to enumerate angle features (extension of PR-S21 pattern, not a new bullet — TF-12 angles are textbook geometry, attribution-grade); **new Savitzky & Golay 1964 bullet added** for TF-8 (this IS a new methodology citation, not just attribution).

## 9a. Lakehouse coordination (consumer-side handshake)

PR-S24 has THREE explicit consumer-contract handoffs to lakehouse, each with a concrete pre-merge action:

1. **`expected_authors` test bump** (lakehouse `test_architecture_md_appendix.py`). New entry: **Savitzky & Golay 1964**. Anzer & Bauer 2021 entry already exists from PR-S21; no change for TF-12. Lakehouse adds the Savitzky & Golay author tuple to `expected_authors` in their pin-bump PR.
2. **Boundary-test column-set update** (lakehouse `src/tests/test_silly_kicks_boundary.py`). The `add_pre_shot_gk_context(actions, *, frames=...)` 6-column surface (vs the 4-column `frames=None` surface) needs the boundary test's `expected_columns` to grow by exactly two: `pre_shot_gk_angle_to_shot_trajectory`, `pre_shot_gk_angle_off_goal_line`. CHANGELOG entry under `### Added` enumerates these by exact name.
3. **Optional opt-in for preprocessing**. Lakehouse can adopt `PreprocessConfig.for_provider("sportec")` etc. to feed cleaner `vx`/`vy`/`speed` into downstream features at any pace — no forced migration. Default `None` = zero behaviour change.
4. **`applyInPandas` schema declaration** (lakehouse only, when adopting preprocessing inside Spark UDFs). Lakehouse `applyInPandas` callers must declare output schemas explicitly via `StructType`. When wrapping `smooth_frames` / `derive_velocities` / `interpolate_frames` inside an `applyInPandas` UDF, the declared `StructType` needs the new schema-extension fields added: `_preprocessed_with: StringType()`, plus whichever of `x_smoothed: DoubleType()`, `y_smoothed: DoubleType()`, `vx: DoubleType()`, `vy: DoubleType()`, `speed: DoubleType()` the consumer surfaces from the UDF. Forgotten schema fields silently drop in `applyInPandas`; lakehouse pin-bump PR should grep their UDF schemas for `_preprocessed_with` to confirm propagation.

Lakehouse pin bump: `silly-kicks>=3.1.0,<4`. No 3.0.x → 3.1.0 migration needed beyond items 1+2 above (item 3 is opt-in adoption; item 4 only applies if item 3 is adopted).

**Internal references:**
- ADR-004 — tracking primitives (linkage + frame-aware compute).
- ADR-005 — frame-aware xfn marker / composition / NOTICE.
- ADR-006 — direction-of-play correctness (PR-S22/S23 erratum).
- PR-S20 spec (`2026-04-30-action-context-pr1-design.md`).
- PR-S21 spec (`2026-05-01-pre-shot-gk-plus-baselines-design.md`).
- PR-S23 spec (`2026-05-02-sk3.0.1-direction-of-play-fix-design.md`).
