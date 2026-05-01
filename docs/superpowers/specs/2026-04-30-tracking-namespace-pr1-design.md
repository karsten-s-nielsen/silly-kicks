# Tracking namespace — PR-1 (primitive layer) — silly-kicks 2.7.0

**Status:** Approved (design)
**Target release:** silly-kicks 2.7.0
**Author:** Karsten S. Nielsen with Claude Opus 4.7 (1M)
**Date:** 2026-04-30
**Predecessor:** 2.6.0 (`docs/superpowers/specs/2026-04-30-pff-fc-events-converter-design.md`)
**Successor:** PR-2 — `action_context()` tracking-aware features (separate spec, separate session, target 2.8.0)
**Triggers:**
- TODO.md `Tracking namespace — silly_kicks.tracking.*` entry deferred from silly-kicks 2.6.0.
- Cross-check session against `karstenskyt__luxury-lakehouse` (Databricks `soccer_analytics.dev_gold.fct_tracking_frames`) confirming 4-provider schema-stress is the right scope and surfacing two field additions (`ball_state`, `team_attacking_direction`) plus two follow-up utility ideas (`infer_ball_carrier`, `sync_score`) for the deferred list.
- Best-practice, long-term TDD/E2E architecture mandate: nothing ships without full test coverage.

---

## 1. Problem

silly-kicks supports six event-data providers (StatsBomb, Opta, Wyscout, Sportec, Metrica, PFF) plus a kloppy event gateway, and provides VAEP / xT / atomic-SPADL action-valuation pipelines on top. Tracking data — the per-frame x/y positions of all players + ball — is **not** a first-class library surface today.

Tracking unlocks analytical capabilities events fundamentally cannot:

- Off-ball player positions: pressure measurement, defensive compactness, pass-option counts, off-ball runs, defender density between shooter and goal.
- Player kinematics: velocity, acceleration, sprint counts, high-intensity distance, fatigue proxies.
- Pre/post-event context windows (e.g., defender behaviour 3 s after a shot).
- Pitch-control / Voronoi / Spearman models — require every player's position at every frame.
- Tracking-aware VAEP: nearest-defender distance at action start, ball-carrier speed, receiver-zone density. Known to materially improve VAEP / xG / xPass calibration (Bekkers 2024; Decroos & Davis 2020).
- Refined GK / pre-shot context — `add_pre_shot_gk_context` becomes genuinely accurate once GK position is known per-frame, not inferred.
- Forward-compatible QA layer: tracking can detect mistagged event locations, supporting silent-drift detection.

Adding tracking touches three correctness frontiers simultaneously: (a) per-provider parser shape (4 providers, 3 native + 1 third-party-derived adapter), (b) cross-provider canonical schema (long-form, 105×68m coordinates, mixed frame-rate handling), (c) action ↔ frames linkage primitive that every downstream feature reduces to. Bundling provider parsers + schema lock + downstream features in one PR conflates failure modes and makes the test surface harder to reason about.

This PR (PR-1) lands **the primitive layer only**: schema + 4 provider adapters + `link_actions_to_frames` + `slice_around_event` + `play_left_to_right` for tracking + ADR-004 charter. No tracking-aware features. PR-2 adds `action_context()` on top of the locked schema in a separate session.

## 2. Goals

1. **First-class `silly_kicks.tracking` namespace** parallel to `silly_kicks.spadl`, with hexagonal pure-function contract: `convert_to_frames(...) -> tuple[pd.DataFrame, TrackingConversionReport]`. Zero I/O; zero global state mutation.
2. **Four-provider coverage** in PR-1: PFF, Sportec/IDSSE, Metrica, SkillCorner. Three adapter codepaths (Sportec native + PFF native + kloppy gateway covering Metrica + SkillCorner). The schema-stress goal — "the canonical schema is locked only after 4 providers load cleanly" — is the load-bearing PR-1 deliverable.
3. **Linkage primitive** (`link_actions_to_frames` + `slice_around_event`) — the single load-bearing cross-pipeline operation that all PR-2+ tracking-aware features build on. Returns pointer DataFrame + LinkReport audit. Tolerance default 0.2 s.
4. **TDD-first**, RED-before-GREEN, 8 implementation loops (preceded by one Loop 0 probe-and-baseline setup step), fully tested. Synthetic fixtures committed; real-data e2e sweep before merge.
5. **Empirical-probe-driven fixture parameterization**: synthetic generators read measured per-provider statistics from a one-off probe of lakehouse + local PFF, written to a committed JSON. Synthetic fixtures are distributional shadows of real data, not hand-crafted from imagination (PR-S18 lesson).
6. **ADR-004** captures the namespace charter so PR-2+ inherits architectural invariants without re-litigation.
7. **Library-grade abstraction.** No lakehouse-shaped assumptions. silly-kicks coordinates are 105×68 m; identifiers follow per-provider events conventions; long-form ball-row encoding. Lakehouse boundary adapters do unit + shape conversion.

## 3. Non-goals

- **No tracking-aware features.** `action_context()`, `pressure_on_carrier()`, pitch-control models, smoothing, multi-frame interpolation, gap-filling, `infer_ball_carrier()`, `sync_score()` — all deferred to follow-up scoping cycles. PR-1 ships primitives only.
- **No raw I/O in the package.** Callers parse provider files (JSON, JSONL.bz2, CSV, XML+positions) into the provider-shaped input DataFrame. The hexagonal contract that all of silly-kicks observes applies to tracking without exception.
- **No streaming / chunked reading.** Tracking files can be large (PFF WC22 single match ≈ 100 MB compressed JSONL), but the library accepts pre-materialized DataFrames. Out-of-core handling is a downstream concern (Spark, Polars-streaming, Dask, etc.).
- **No StatsBomb 120×80 coordinate units.** The lakehouse uses StatsBomb units for legacy reasons; silly-kicks tracking is 105×68 m, matching `silly_kicks.spadl`. Boundary adapters in lakehouse pipelines do unit conversion.
- **No wide-form schema.** Long-form (one row per (frame, player), ball as own row with `is_ball=True`) is canonical. Wide-form pivots happen at consumer boundaries.
- **No PR-2 work bundled in.** `action_context()` and friends ship in their own spec/plan/cycle against the locked schema.
- **No env-var-gated tests in CI.** Synthetic fixtures cover unit + integration. The real-data sweep (`tests/test_tracking_real_data_sweep.py`) is `e2e`-marked and skips with explicit reason on missing local data.
- **No ReSpo.Vision adapter.** Pending licensing — flagged in TODO.md.

## 4. Design

### 4.1 Architecture & module layout

The tracking namespace mirrors the SPADL events layout, adapted for the asymmetric kloppy-tracking coverage:

```
silly_kicks/tracking/
├── __init__.py              # public exports + lazy kloppy import (try/except ImportError)
├── schema.py                # TRACKING_FRAMES_COLUMNS, per-provider variants,
│                            # TRACKING_CONSTRAINTS, TRACKING_CATEGORICAL_DOMAINS,
│                            # TrackingConversionReport, LinkReport
├── sportec.py               # native — DFL Position-XML shaped input
├── pff.py                   # native — PFF JSONL-flattened-frames input
├── kloppy.py                # gateway — Metrica + SkillCorner via kloppy 3.18
├── utils.py                 # link_actions_to_frames, slice_around_event,
│                            # play_left_to_right (tracking variant), _derive_speed
└── _direction.py            # SHARED with spadl/_direction (refactor PR-S18 helper
                             # out of pff.py into a tracking-shareable home)
```

**Three adapter codepaths, four providers:**

| Provider     | Adapter                  | Module                            | Input shape                                              |
|--------------|--------------------------|-----------------------------------|----------------------------------------------------------|
| Sportec/IDSSE| native                   | `silly_kicks/tracking/sportec.py` | DataFrame parsed from DFL Position XML by caller         |
| PFF          | native                   | `silly_kicks/tracking/pff.py`     | DataFrame parsed from PFF `.jsonl.bz2` by caller         |
| Metrica      | kloppy gateway           | `silly_kicks/tracking/kloppy.py`  | `kloppy.TrackingDataset` from `kloppy.metrica.load_tracking_csv` |
| SkillCorner  | kloppy gateway           | `silly_kicks/tracking/kloppy.py`  | `kloppy.TrackingDataset` from `kloppy.skillcorner.load_tracking` |

**Why the asymmetry.** As of kloppy 3.18.0, kloppy ships tracking parsers for PFF, Metrica, SkillCorner, SecondSpectrum, Signality, StatsPerform — but **not** Sportec/IDSSE tracking. Sportec must be native. PFF could go through the kloppy gateway, but a native module is preferred for two reasons: (1) symmetry with `silly_kicks.spadl.pff` (PR-S18) — both PFF adapters share the `_direction.py` helper for `home_team_start_left[_extratime]` perspective handling; (2) failure isolation — kloppy's PFF tracking parser is recently-added and a kloppy regression should not propagate into PFF, our highest-stakes provider (WC 2022 dataset).

**The `_direction.py` extraction is bundled into PR-1.** PR-S18's direction helper currently lives inside `silly_kicks/spadl/pff.py`. It belongs in a shared location so both events and tracking PFF adapters consume one implementation. The extraction is a strict refactor (move + import update); zero behaviour change in events. National Park Principle bundle, ~10 lines.

**Public-API surface (`silly_kicks/tracking/__init__.py`):**

```python
__all__ = [
    "TRACKING_FRAMES_COLUMNS",
    "TRACKING_CONSTRAINTS",
    "TrackingConversionReport",
    "LinkReport",
    "kloppy",
    "link_actions_to_frames",
    "play_left_to_right",
    "pff",
    "schema",
    "slice_around_event",
    "sportec",
    "utils",
]
```

The `kloppy` submodule is conditionally imported with `try/except ImportError` (mirrors `silly_kicks.spadl.__init__.py:49-52`).

### 4.2 Schema (`silly_kicks/tracking/schema.py`)

#### `TRACKING_FRAMES_COLUMNS` — 19 columns, long-form

```python
TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    "game_id":                  "int64",     # provider-dependent dtype variant below
    "period_id":                "int64",     # 1, 2, 3=ET1, 4=ET2, 5=PSO
    "frame_id":                 "int64",     # frame ordinal within period
    "time_seconds":             "float64",   # period-relative seconds
    "frame_rate":               "float64",   # Hz; mixed across providers (10 / 25 / 30)
    "player_id":                "int64",     # NaN for ball rows; provider-dependent variant
    "team_id":                  "int64",     # NaN for ball rows; provider-dependent variant
    "is_ball":                  "bool",
    "is_goalkeeper":            "bool",      # False for ball rows
    "x":                        "float64",   # SPADL meters [0, 105]
    "y":                        "float64",   # SPADL meters [0, 68]
    "z":                        "float64",   # NaN where provider lacks z (most player rows)
    "speed":                    "float64",   # m/s; populated for all rows (hybrid policy)
    "speed_source":             "object",    # "native" | "derived" | NaN
    "ball_state":               "object",    # "alive" | "dead"; NaN where unavailable
    "team_attacking_direction": "object",    # "ltr" | "rtl"; NaN for ball rows
    "confidence":               "object",    # provider-specific tier
    "visibility":               "object",    # provider-specific tier
    "source_provider":          "object",    # "pff" | "sportec" | "metrica" | "skillcorner"
}

# Per-provider dtype variants (mirrors KLOPPY_SPADL_COLUMNS / PFF_SPADL_COLUMNS):
KLOPPY_TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    **TRACKING_FRAMES_COLUMNS,
    "game_id": "object", "player_id": "object", "team_id": "object",
}

SPORTEC_TRACKING_FRAMES_COLUMNS: dict[str, str] = KLOPPY_TRACKING_FRAMES_COLUMNS
"""Sportec native uses DFL string identifiers — same dtype shape as kloppy."""

PFF_TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    **TRACKING_FRAMES_COLUMNS,
    "player_id": "Int64", "team_id": "Int64",
}
"""PFF: nullable Int64 to support NaN on ball rows (PR-S18 events convention)."""
```

#### `TRACKING_CONSTRAINTS`

```python
TRACKING_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "period_id":    (1, 5),
    "time_seconds": (0, float("inf")),
    "frame_rate":   (1, 60),         # rejects 0 Hz and unrealistic >60 Hz
    "frame_id":     (0, float("inf")),
    "x":            (0, 105.0),
    "y":            (0, 68.0),
    "z":            (0, 10.0),       # ball arc realistic upper bound; NaN allowed
    "speed":        (0, 50.0),       # m/s; covers ball speed; player rows naturally <15
}
```

#### `TRACKING_CATEGORICAL_DOMAINS`

```python
TRACKING_CATEGORICAL_DOMAINS: dict[str, frozenset[str]] = {
    "ball_state":               frozenset({"alive", "dead"}),
    "team_attacking_direction": frozenset({"ltr", "rtl"}),
    "speed_source":             frozenset({"native", "derived"}),
    "source_provider":          frozenset({"pff", "sportec", "metrica", "skillcorner"}),
}
```

#### `TrackingConversionReport`

```python
@dataclasses.dataclass(frozen=True)
class TrackingConversionReport:
    """Audit trail for tracking convert_to_frames(). Frame-shaped, not event-shaped."""
    provider: str
    total_input_frames: int
    total_output_rows: int                          # long-form expansion factor
    n_periods: int
    frame_coverage_per_period: dict[int, float]    # period_id -> fraction of expected frames present
    ball_out_seconds_per_period: dict[int, float]  # sum of ball_state="dead" durations
    nan_rate_per_column: dict[str, float]          # column name -> fraction of NaN rows
    derived_speed_rows: int                         # how many rows had speed derived
    unrecognized_player_ids: set[str | int]         # IDs in input not resolvable via roster

    @property
    def has_unrecognized(self) -> bool:
        return len(self.unrecognized_player_ids) > 0
```

#### `LinkReport`

```python
@dataclasses.dataclass(frozen=True)
class LinkReport:
    """Audit trail for link_actions_to_frames()."""
    n_actions_in: int
    n_actions_linked: int
    n_actions_unlinked: int                   # frame_id NaN
    n_actions_multi_candidate: int            # >1 frame within tolerance — closest used
    per_provider_link_rate: dict[str, float]  # source_provider -> linked / in
    max_time_offset_seconds: float            # max |Δt| among linked rows
    tolerance_seconds: float                  # echoes the call argument

    @property
    def link_rate(self) -> float:
        return self.n_actions_linked / max(self.n_actions_in, 1)
```

### 4.3 Public API & data flow

#### Three converter signatures

**`silly_kicks.tracking.sportec.convert_to_frames`** (native, DFL Position-XML shaped):
```python
def convert_to_frames(
    raw_frames: pd.DataFrame,
    home_team_id: str,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """SPORTEC_TRACKING_FRAMES_COLUMNS-shaped output, 105×68 m SPADL coordinates."""
```

**`silly_kicks.tracking.pff.convert_to_frames`** (native, JSONL-flattened):
```python
def convert_to_frames(
    raw_frames: pd.DataFrame,
    home_team_id: int,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """PFF_TRACKING_FRAMES_COLUMNS-shaped output, 105×68 m SPADL coordinates.
    Reuses _direction.flip_per_period_to_ltr (extracted from spadl/pff.py)."""
```

**`silly_kicks.tracking.kloppy.convert_to_frames`** (gateway):
```python
def convert_to_frames(
    dataset: kloppy.domain.TrackingDataset,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, TrackingConversionReport]:
    """KLOPPY_TRACKING_FRAMES_COLUMNS-shaped output. Dispatches on
    dataset.metadata.provider (Provider.METRICA / Provider.SKILLCORNER).
    Provider.PFF and Provider.SPORTEC raise NotImplementedError — route
    through silly_kicks.tracking.pff and silly_kicks.tracking.sportec."""
```

The kloppy gateway issues `dataset.transform(to_pitch_dimensions=MetricPitchDimensions(105, 68), to_orientation=Orientation.HOME_AWAY)` once before materializing rows.

#### Two linkage utilities (`silly_kicks.tracking.utils`)

```python
def link_actions_to_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    tolerance_seconds: float = 0.2,
) -> tuple[pd.DataFrame, LinkReport]:
    """For each action, find the single nearest frame by (period_id, time_seconds).

    Returns:
        pointer DataFrame with columns:
            action_id           int64
            frame_id            Int64       NaN if no frame within tolerance
            time_offset_seconds float64     signed; NaN where unlinked
            n_candidate_frames  int64       number of distinct frame_ids within tolerance
            link_quality_score  float64     1.0 - |Δt|/tolerance_seconds; NaN where unlinked
        LinkReport audit.

    Implementation: sort frames by (period_id, time_seconds), drop_duplicates
    on frame_id (long-form has multiple rows per frame; we link to the frame,
    not a specific player-row), then merge_asof with direction='nearest',
    tolerance.
    """

def slice_around_event(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    pre_seconds: float = 0.0,
    post_seconds: float = 0.0,
) -> pd.DataFrame:
    """For each action, return all frames within [t - pre_seconds, t + post_seconds]
    in the same period.

    Output: long-form frames slice with action_id and time_offset_seconds joined.
    One row per (action_id, frame_id, player_id_or_ball).

    Implementation: pd.merge actions × frames on period_id, filter on time
    window. Vectorized; no Python loop. Window does not cross periods.
    """
```

#### `play_left_to_right` for tracking

```python
def play_left_to_right(frames: pd.DataFrame, home_team_id: int | str) -> pd.DataFrame:
    """Mirror x and y for periods where the home team did not start left.

    Differences from spadl.utils.play_left_to_right:
    1. Operates on long-form rows (player rows AND ball rows).
    2. After flipping, sets team_attacking_direction = "ltr" for all flipped rows.
    3. ball_state, frame_id, time_seconds, etc. unchanged; only x/y mirror.

    Vectorized: x_flipped = 105.0 - x, y_flipped = 68.0 - y, applied per-period
    via np.where on a flip-mask.
    """
```

#### `_derive_speed` (private helper in `utils.py`)

```python
def _derive_speed(frames: pd.DataFrame) -> pd.DataFrame:
    """Compute speed = sqrt((Δx)² + (Δy)²) * frame_rate per (player_id, period_id) group.

    Used by Metrica + SkillCorner adapters where the provider doesn't supply speed.
    Edge cases:
      - First frame of each (player, period): speed = NaN
      - Period boundary: groupby on (player_id, period_id) prevents cross-period leakage
      - Ball treated as one logical "player"
      - Players exiting frame: NaN row break, downstream diff produces NaN
    """
```

#### Coordinate normalization — per-adapter responsibility

| Provider     | Source coords                                            | Transformation                                      |
|--------------|----------------------------------------------------------|------------------------------------------------------|
| Sportec      | DFL Position XML: pitch-centered meters, ~[-52.5, 52.5] × ~[-34, 34] | `x_spadl = source_x + 52.5`, `y_spadl = source_y + 34`, then per-period direction flip |
| PFF          | PFF JSON: pitch-centered meters, ~[-52.5, 52.5] × ~[-34, 34] | Same as sportec; reuses `_direction.flip_per_period_to_ltr` extracted from PR-S18 |
| Metrica      | kloppy normalizes from native [0, 1] | `dataset.transform(to_pitch_dimensions=MetricPitchDimensions(105, 68), to_orientation=Orientation.HOME_AWAY)` |
| SkillCorner  | kloppy normalizes from broadcast pitch coords | Same kloppy transform call |

#### Linkage internal-dedup decision

`link_actions_to_frames` accepts the long-form frames DataFrame (multiple rows per `frame_id`) and **internally deduplicates on `(period_id, frame_id)`** before `merge_asof`. Documented in the docstring. Caller does not pre-dedup.

#### `slice_around_event` output filtering

PR-1 returns the full long-form slice (every player row). No `players_only` flag. YAGNI; PR-2 features will signal whether a builtin filter is worth it.

### 4.4 ADR-004 — silly_kicks.tracking namespace charter

Lives at `docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md`. Mirrors ADR-001/002/003 in shape; ~150 lines. Captures the architectural invariants so PR-2+ inherits them without re-litigation.

```
Status: Accepted (silly-kicks 2.7.0)
Date: 2026-04-30
Drivers: TODO.md tracking-namespace entry (silly-kicks 2.6.0); lakehouse cross-check
         session (4-provider schema-stress).

Decision: silly_kicks.tracking is a first-class namespace parallel to
silly_kicks.spadl, with the following nine invariants:

1. Hexagonal pure-function contract. Zero I/O. Zero global state mutation.

2. Canonical 19-column long-form schema (TRACKING_FRAMES_COLUMNS).
   Per-provider dtype variants (KLOPPY/SPORTEC/PFF) follow the events
   precedent; identifier dtypes match the same provider's SPADL converter
   (ADR-001 cross-namespace consistency).

3. SPADL 105 × 68 m coordinates, bottom-left origin. The lakehouse's
   StatsBomb 120 × 80 unit choice is explicitly NOT reimported; boundary
   adapters in downstream pipelines do unit conversion.

4. Adapter taxonomy: native modules for Sportec and PFF; kloppy gateway
   for Metrica + SkillCorner. PFF native is preferred over kloppy's PFF
   tracking parser for symmetry with silly_kicks.spadl.pff and to share
   the _direction helper extracted from PR-S18.

5. Long-form ball-row encoding: is_ball=True row per frame; player_id /
   team_id NaN on ball rows. Wide-form pivots happen at consumer
   boundaries, not in the library.

6. Linkage primitive (utils.link_actions_to_frames +
   utils.slice_around_event) returns a pointer DataFrame plus LinkReport
   audit. Tolerance default 0.2 s. NaN frame_id on no-link. This is THE
   load-bearing cross-pipeline operation that all PR-2+ tracking-aware
   features build on.

7. Speed populated for all providers (hybrid: trust native, derive
   where missing). speed_source provenance column ("native" / "derived"
   / NaN) records origin. Adapters compute derivation; no downstream
   consumer should re-derive.

8. Synthetic-only committed fixtures, distributionally parameterized
   from a one-off empirical probe of lakehouse + local PFF tracking
   (committed JSON baseline). The probe script and JSON are both
   committed; the real datasets are not. L3b e2e-marked real-data sweep
   runs before each tracking PR's single commit.

9. Tracking-aware features explicitly DEFERRED to follow-up scoping
   cycles, with this priority order (per lakehouse session):
   (i)   action_context() — tracking-aware VAEP/xG features
   (ii)  pressure_on_carrier()
   (iii) infer_ball_carrier() — utility, lakehouse session addition
   (iv)  sync_score() — QA primitive, lakehouse session addition
   (v)   pitch-control models (Spearman / Voronoi)
   (vi)  smoothing primitives (Savitzky-Golay, EMA)
   (vii) multi-frame interpolation / gap filling
   (viii) ReSpo.Vision adapter (licensing-gated, not engineering-gated)
   PR-1 ships primitives only.

Consequences:
 - Future PR-2 builds on locked schema + linkage primitives without re-ratifying
 - Lakehouse boundary stays one-direction-of-conversion (long-form -> wide-form,
   105x68 -> 120x80, object id -> bigint id), no library changes needed
 - Adding a new tracking provider = new adapter module + new fixture
   generator + new probe entry; schema is invariant
```

## 5. Testing & validation

### 5.1 Test file layout

```
tests/datasets/tracking/
├── empirical_probe_baselines.json       # one-off probe output, committed
├── pff/
│   ├── generate_synthetic.py            # generator script (committed)
│   ├── tiny.parquet                     # ~3 s × 22 players ~1.6k rows
│   └── medium_halftime.parquet          # ~60 s spanning HT crossover ~33k rows
├── sportec/  {generate_synthetic.py, tiny.parquet, medium_halftime.parquet}
├── metrica/  {generate_synthetic.py, tiny.parquet, medium_halftime.parquet}
└── skillcorner/ {generate_synthetic.py, tiny.parquet, medium_halftime.parquet}

scripts/
└── probe_tracking_baselines.py           # one-off, run during PR-1 development

tests/
├── test_tracking_schema.py
├── test_tracking_sportec.py
├── test_tracking_pff.py
├── test_tracking_kloppy.py
├── test_tracking_utils_link.py
├── test_tracking_utils_slice.py
├── test_tracking_utils_play_left_to_right.py
├── test_tracking_utils_derive_speed.py
├── test_tracking_cross_provider_parity.py
├── test_tracking_real_data_sweep.py     # marked e2e
└── test_public_api_examples.py          # auto-discovers tracking modules
```

### 5.2 Empirical-probe-driven fixture parameterization

`scripts/probe_tracking_baselines.py` is a one-off script run once during PR-1 development. Output: `tests/datasets/tracking/empirical_probe_baselines.json` (committed). Both committed.

**Probe Source 1 — Lakehouse (Databricks SQL):** `soccer_analytics.dev_gold.fct_tracking_frames`. Providers: metrica, idsse (sportec), skillcorner. Per-provider stats extracted: `frame_rate_hz`, period_id distribution + per-period frame counts, `ball_state` alive/dead duration histogram, NaN-rate-per-column, identifier dtype, x/y/z and speed percentiles (p1, p25, p50, p75, p99), per-frame player count, ball-row presence rate, period-boundary frame-coverage timing.

**Probe Source 2 — Local PFF WC2022 JSONL.bz2:** `D:\[Karsten]\Dropbox\[Microsoft]\Downloads\FIFA World Cup 2022\Tracking Data\*.jsonl.bz2` (65 matches; 1 match minimum to characterize). Same statistics extracted from raw JSON.

**JSON shape:**

```json
{
  "probe_run_date": "2026-04-30",
  "probe_run_source_lakehouse_table": "soccer_analytics.dev_gold.fct_tracking_frames",
  "probe_run_source_pff_path_marker": "<local-path-marker>",
  "providers": {
    "pff":         { "frame_rate_hz": 30, "...": "..." },
    "sportec":     { "frame_rate_hz": 25, "...": "..." },
    "metrica":     { "frame_rate_hz": 25, "speed_native_supplied": false, "...": "..." },
    "skillcorner": { "frame_rate_hz": 10, "...": "..." }
  }
}
```

**How synthetic generators consume it:** each `generate_synthetic.py` imports `empirical_probe_baselines.json` and reads the per-provider dict. Frame rate, player-count mode, ball-out duty cycle, NaN-rate signatures, and period-boundary timing all come from measured data. The synthetic motion itself is deterministic (uniform / piecewise-linear) and does not depend on real data; only the **structural parameters** are calibrated.

**Drift-detection in L3b sweep:** the real-data sweep re-computes the same statistics from local data and asserts agreement with the committed JSON within tolerance. Catches both lakehouse mart drift and synthetic-fixture drift.

### 5.3 Synthetic fixture sizes

- **`tiny.parquet`** — 3 s × 22 players + ball, deterministic uniform motion at known speeds, single period. ~1.6k rows. Used for: exact derivation correctness (`_derive_speed` analytic match), exact coordinate transforms, exact ball-row bookkeeping. Tests run <50 ms.
- **`medium_halftime.parquet`** — 60 s spanning end-of-period-1 through start-of-period-2, 22 players + ball, with realistic ball-out interval and period-flip. ~33k rows. Used for: period-boundary edge cases, `home_team_start_left` direction normalization, `ball_state="dead"` transitions, mixed-Hz parity.

Per-provider parquet ≈ 200 KB committed; total ≈ 800 KB across 4 providers (within `tests/datasets/{idsse,metrica}/` parquet precedent).

### 5.4 Unit test scope per file

| File | What it asserts |
|------|------------------|
| `test_tracking_schema.py` | `TRACKING_FRAMES_COLUMNS` is exactly 19 columns. Per-provider variants are strict supersets-by-dtype-override. `TRACKING_CATEGORICAL_DOMAINS` keys are a subset of `TRACKING_FRAMES_COLUMNS` keys. `TrackingConversionReport` and `LinkReport` are frozen dataclasses. `LinkReport.link_rate` arithmetic on edge cases (n_actions_in=0 → returns 0.0, not ZeroDivisionError). |
| `test_tracking_sportec.py` / `test_tracking_pff.py` / `test_tracking_kloppy.py` | (1) `convert_to_frames(tiny)` produces correctly-shaped output with exact dtype match. (2) Output columns are exactly the schema column set. (3) Coordinate bounds: `x ∈ [0, 105]`, `y ∈ [0, 68]`. (4) Ball rows: `is_ball=True`, `player_id=NaN`, `team_id=NaN`, `is_goalkeeper=False`, `team_attacking_direction=NaN`. (5) `TrackingConversionReport.total_input_frames` matches input. (6) `home_team_start_left=False` flips x relative to `=True` baseline (asserted on tiny). (7) `medium_halftime` exercises period-flip — period-2 frames have post-flip coordinates consistent with period-1. (8) `unrecognized_player_ids` populated on synthetic input that includes a roster gap. |
| `test_tracking_utils_link.py` | Exact-time-match action → `time_offset_seconds=0`, `link_quality_score=1.0`. Action 0.15 s off → linked at tolerance=0.2, unlinked at tolerance=0.1. Empty actions → empty pointer DataFrame, valid LinkReport with `n_actions_in=0`. Action with no frame in any period → `frame_id=NaN`, `LinkReport.n_actions_unlinked=1`. Multi-candidate (2 frames within tolerance) → closest one chosen, `n_candidate_frames=2`. Cross-period: action in period 1, only frames in period 2 → unlinked. Test-default-stability: tolerance=0.2 default validated by an explicit-default test. |
| `test_tracking_utils_slice.py` | `pre=0, post=0` returns the same set of `frame_id`s `link_actions_to_frames` would link to (consistency between the two utilities). `pre=0.5, post=0.5` returns ~25 frames at 25 Hz × 22 players + ball ≈ 575 rows per action. Action near period boundary: window does NOT cross periods. Empty intersection → empty result, no error. |
| `test_tracking_utils_play_left_to_right.py` | Ball row x flipped: `home_team_start_left=False, period=1, ball_x=20 → 85`. Player rows flip identically to ball. Already-LTR frames pass through unchanged. `team_attacking_direction` post-flip = "ltr" for all flipped rows; original "rtl" rows updated. NaN x/y stays NaN. |
| `test_tracking_utils_derive_speed.py` | Uniform motion 1 m/s → derived speed = 1.0 ± 1e-6. First frame per (player, period) → NaN. Period boundary: speed at first frame of period 2 = NaN, not derived from last frame of period 1. Ball treated as one logical entity. |

### 5.5 Cross-provider parity gate (committed)

```python
@pytest.mark.parametrize("provider", ["pff", "sportec", "metrica", "skillcorner"])
def test_tracking_bounds(provider): ...      # all rows pass TRACKING_CONSTRAINTS

@pytest.mark.parametrize("provider", ["pff", "sportec", "metrica", "skillcorner"])
def test_tracking_link_rate(provider): ...    # synthetic events ↔ frames link rate ≥ 0.95

def test_tracking_distance_to_ball_distribution_overlap():
    """Per-provider distance-to-ball percentiles (p25, p50, p75, p95) on
    medium_halftime fixture must overlap within tolerance — proves coordinate
    normalization is consistent across providers (the schema-stress goal)."""
```

The 0.95 link-rate threshold is for synthetic fixtures where actions are constructed to fall on real frame times; lower thresholds for L3b real data are tracked separately in the sweep summary.

### 5.6 Real-data sweep (e2e-marked)

```python
@pytest.mark.e2e
@pytest.mark.parametrize("provider,env_var", [
    ("pff",         "PFF_TRACKING_DIR"),
    ("sportec",     "IDSSE_TRACKING_DIR"),
    ("metrica",     "METRICA_TRACKING_DIR"),
    ("skillcorner", "SKILLCORNER_TRACKING_DIR"),
])
def test_tracking_real_data_sweep(provider, env_var):
    path = os.environ.get(env_var)
    if not path:
        pytest.skip(f"{env_var} not set; skipping {provider} real-data sweep")
    # Per-provider full-match sweep:
    #   - dtype audit
    #   - bounds audit (TRACKING_CONSTRAINTS)
    #   - NaN-rate-per-column audit
    #   - distance-to-ball percentile baseline emit
    #   - linkage rate against same-match real events (when available)
    # Asserts: probe-baseline JSON match within tolerance.
    # Output: structured JSON summary written to stdout + markdown rendering for PR description.
```

`pytest.skip` with explicit reason — never silent-pass-no-work (memory: silently-skipping-tests-hide-breakage).

### 5.7 Public-API Examples coverage

Every public def in `silly_kicks/tracking/{__init__, schema, utils, sportec, pff, kloppy}.py` gets an `Examples` section in its docstring. The existing `tests/test_public_api_examples.py` auto-discovers the new modules and enforces this — no new test file needed.

### 5.8 Pre-commit gates

- `ruff` + `pyright` + `pytest -m "not e2e"` (Shift Left).
- `tests/test_tracking_real_data_sweep.py` run locally with all four `*_TRACKING_DIR` env vars set.
- `/final-review` (mandatory per memory; mad-scientist-skills:final-review).
- One commit per branch (squash merge). `Co-Authored-By: Claude Opus 4.7 (1M context)` trailer.

### 5.9 TDD ordering (8 RED-GREEN-REFACTOR loops)

The implementation plan (drafted by `superpowers:writing-plans` next) sequences work as:

```
Loop 0 (one-off, scoped to this PR):
  - Write & run scripts/probe_tracking_baselines.py
  - Output: tests/datasets/tracking/empirical_probe_baselines.json (committed)
  - All synthetic generators import this JSON

Loop 1: Schema constants + TrackingConversionReport + LinkReport
        (test_tracking_schema.py)

Loop 2: _derive_speed (test_tracking_utils_derive_speed.py)

Loop 3: play_left_to_right for tracking
        (test_tracking_utils_play_left_to_right.py)

Loop 4: link_actions_to_frames (test_tracking_utils_link.py)

Loop 5: slice_around_event (test_tracking_utils_slice.py)

Loop 6: Sportec native adapter (test_tracking_sportec.py)
        Includes synthetic generator using empirical baseline.

Loop 7: PFF native adapter (test_tracking_pff.py)
        Includes _direction.py extraction from spadl/pff.py (refactor;
        zero behaviour change in events).

Loop 8: Kloppy gateway (test_tracking_kloppy.py)

Post-loops:
  - Cross-provider parity gate (L2b) becomes meaningful — runs against
    all 4 providers' fixtures.
  - Real-data sweep (L3b) runs locally; summary pasted in PR description.
  - /final-review.
  - Single commit; squash-merge.
```

## 6. Branch / version strategy

- **Branch:** `feat/tracking-namespace-pr1`.
- **Target version:** silly-kicks **2.7.0** (minor bump — additive new namespace, no API breakage in `silly_kicks.spadl` or `silly_kicks.vaep`).
- **Single commit per branch** (memory: commit policy). Squash-merge on GitHub.
- **PR-2** (`action_context`, separate session) targets **2.8.0** against locked 2.7.0 schema.

## 7. Lakehouse boundary impact (informational)

PR-1 lands a long-form, 105×68 m, object/Int64-identifier-typed, 19-column schema. The lakehouse `fct_tracking_frames` mart is wide-form, StatsBomb 120×80 unit, bigint-identifier-typed, 15-column.

The lakehouse boundary adapter is one direction of conversion — long-form → wide-form pivot, 105×68 → 120×80 unit conversion, dtype harmonization. This PR does not attempt to satisfy both shapes; the lakehouse boundary is the lakehouse's responsibility (per ADR-004 invariant 3).

PR-2's `action_context()` will benefit lakehouse consumers `fct_action_values`, `fct_xg_predictions_v2`, `fct_pass_timing`, `fct_shots` (per lakehouse session). PR-1 alone unlocks no new lakehouse marts; it unlocks the *capability* for PR-2+ to do so.

## 8. TODO.md updates

PR-1 closes the existing TODO.md `Tracking namespace — silly_kicks.tracking.*` entry. New deferred items added (priority order from ADR-004 invariant 9):

1. **`action_context()`** — tracking-aware VAEP/xG features (PR-2, separate session).
2. **`pressure_on_carrier()`** — pressure on ball-carrier feature.
3. **`infer_ball_carrier()`** — heuristic ball-carrier-per-frame inference (lakehouse session addition).
4. **`sync_score()`** — per-action tracking↔events sync-quality score (lakehouse session addition).
5. **Pitch-control models** (Spearman / Voronoi).
6. **Smoothing primitives** (Savitzky-Golay, EMA).
7. **Multi-frame interpolation / gap filling.**
8. **ReSpo.Vision adapter** (pending licensing).

## 9. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| kloppy 3.18 tracking parsers have edge cases (e.g., metrica_csv ball_state field, skillcorner pre-period flip) | L3b real-data sweep exercises kloppy on real Metrica + SkillCorner samples; bugs surface before merge. Native PFF + Sportec isolate failures from kloppy regressions. |
| Empirical probe results drift across lakehouse mart updates | L3b sweep re-computes probe stats and asserts JSON-baseline agreement. Catches drift in either direction. |
| Synthetic fixtures pass but real data fails | L3b is the pre-merge gate. The empirical-probe-driven generator design specifically reduces this risk by making fixtures distributional shadows of real data. |
| `link_actions_to_frames` 0.2 s default surprises users on broadcast tracking (10 Hz SkillCorner) | 0.2 s = 2 frames at 10 Hz, still a valid link window; users with looser requirements override per-call. Documented in docstring. |
| Schema lock too tight (a 5th provider can't fit) | 5th provider out of PR-1 scope; the four chosen cover JSON / JSONL / CSV / XML and 10/25/30 Hz, the prevailing heterogeneity. ADR-004 invariant 2 allows new per-provider dtype variants without schema change. |
| _direction.py extraction breaks PR-S18 events PFF behaviour | Pure refactor (move + import update); event tests must pass unchanged before tracking work begins. |

## 10. References

- `TODO.md` — `Tracking namespace — silly_kicks.tracking.*` entry (silly-kicks 2.6.0).
- `docs/superpowers/specs/2026-04-30-pff-fc-events-converter-design.md` — PR-S18 events PFF design (predecessor).
- `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md` — identifier convention precedent.
- Bekkers, J. (2024). *DEFCON-style pressure metrics from tracking data*.
- Decroos, T. & Davis, J. (2020). *Player Vectors* / VAEP with tracking-aware features.
- Lakehouse cross-check session (2026-04-30): `karstenskyt__luxury-lakehouse` `fct_tracking_frames` mart — 38M rows, 3 providers, 20 matches.
