# ADR-004: silly_kicks.tracking namespace charter

| Field | Value |
|---|---|
| **Date** | 2026-04-30 |
| **Status** | Accepted (silly-kicks 2.7.0) |
| **Deciders** | Karsten S. Nielsen, Claude Opus 4.7 (1M) |

## Context

silly-kicks supports six event-data providers (StatsBomb, Opta, Wyscout,
Sportec, Metrica, PFF) plus a kloppy event gateway, and provides VAEP / xT
/ atomic-SPADL action-valuation pipelines on top. Tracking data — the
per-frame x/y positions of all players + ball — is **not** a first-class
library surface today.

Tracking unlocks analytical capabilities events fundamentally cannot:
off-ball player positions, per-frame velocity / acceleration, pre/post-event
context windows, pitch-control / Voronoi / Spearman models, and refined
GK / pre-shot context. Tracking-aware VAEP is known to materially improve
calibration (Bekkers 2024; Decroos & Davis 2020).

This ADR captures the namespace charter so PR-2+ inherits architectural
invariants without re-litigation. PR-1 (`feat/tracking-namespace-pr1`,
silly-kicks 2.7.0) lands the primitive layer only: schema + 4 provider
adapters + linkage primitive. Tracking-aware features (`action_context`,
`pressure_on_carrier`, pitch control, etc.) are explicitly DEFERRED to
follow-up scoping cycles.

## Decision

`silly_kicks.tracking` is a first-class namespace parallel to
`silly_kicks.spadl`, with the following nine invariants:

### 1. Hexagonal pure-function contract

`convert_to_frames(...) -> tuple[pd.DataFrame, TrackingConversionReport]`.
Zero I/O, zero global state mutation. Identical contract to the events
namespace (per ADR-001 cross-namespace consistency).

### 2. Canonical 19-column long-form schema

`TRACKING_FRAMES_COLUMNS` — 19 columns; one row per (frame, player); ball
as own row with `is_ball=True`. Per-provider dtype variants
(`KLOPPY_TRACKING_FRAMES_COLUMNS`, `SPORTEC_TRACKING_FRAMES_COLUMNS`,
`PFF_TRACKING_FRAMES_COLUMNS`) follow the events precedent; identifier
dtypes match the same provider's SPADL converter (per ADR-001).

The schema additionally carries:

- `ball_state`: `"alive" | "dead"`; NaN where unavailable.
- `team_attacking_direction`: `"ltr" | "rtl"`; NaN for ball rows.
- `speed_source`: `"native" | "derived"` provenance column.
- `source_provider`: `"pff" | "sportec" | "metrica" | "skillcorner"`.

### 3. SPADL 105 × 68 m coordinates, bottom-left origin

The lakehouse's StatsBomb 120 × 80 unit choice is explicitly NOT
reimported. silly-kicks tracking is metric, matching `silly_kicks.spadl`.
Boundary adapters in downstream pipelines do unit conversion. The
empirical probe (see invariant 8) records the lakehouse mart as
`statsbomb_120x80` so synthetic fixtures cannot accidentally inherit
the unit choice.

### 4. Adapter taxonomy

Native modules for Sportec and PFF; kloppy gateway for Metrica +
SkillCorner.

| Provider     | Adapter         | Module                            |
|--------------|------------------|-----------------------------------|
| Sportec/IDSSE| native           | `silly_kicks/tracking/sportec.py` |
| PFF          | native           | `silly_kicks/tracking/pff.py`     |
| Metrica      | kloppy gateway   | `silly_kicks/tracking/kloppy.py`  |
| SkillCorner  | kloppy gateway   | `silly_kicks/tracking/kloppy.py`  |

PFF native is preferred over kloppy's PFF tracking parser for symmetry
with `silly_kicks.spadl.pff` (PR-S18) and to share the `_direction`
helper extracted from PR-S18. Sportec must be native because kloppy
3.18 has no Sportec/IDSSE tracking parser. The kloppy gateway raises
`NotImplementedError` for `Provider.PFF` and `Provider.SPORTEC` —
callers route through the native adapters.

### 5. Long-form ball-row encoding

`is_ball=True` row per frame; `player_id` / `team_id` NaN on ball rows.
Wide-form pivots happen at consumer boundaries, not in the library.

### 6. Linkage primitive

`utils.link_actions_to_frames(actions, frames, tolerance_seconds=0.2)`
returns a pointer DataFrame plus a `LinkReport` audit. NaN `frame_id`
on no-link. `utils.slice_around_event` returns a long-form windowed
slice. These two utilities are THE load-bearing cross-pipeline operation
that all PR-2+ tracking-aware features build on.

The default tolerance of 0.2 s is pinned by an explicit
default-stability test (memory: tests-crossing-pipelines-need-default-
stable-params). Changing the default requires updating the test.

### 7. Hybrid speed policy

Speed populated for all providers. Trust native (PFF, Sportec); derive
where missing (Metrica, SkillCorner) via `utils._derive_speed`. The
`speed_source` column records provenance (`"native" | "derived" | NaN`).
Adapters compute derivation; no downstream consumer should re-derive.

### 8. Synthetic-only committed fixtures, empirical-probe parameterized

Per-provider synthetic fixtures (`tests/datasets/tracking/{provider}/
{tiny,medium_halftime}.parquet`) are distributionally parameterized
from a one-off empirical probe of lakehouse + local PFF tracking
(`scripts/probe_tracking_baselines.py` and
`tests/datasets/tracking/empirical_probe_baselines.json` — both
committed). The real datasets are NOT committed.

The L3b real-data sweep (`tests/test_tracking_real_data_sweep.py`,
`e2e`-marked) re-computes the same statistics from local data and
asserts agreement with the committed JSON within tolerance. This
catches both lakehouse mart drift and synthetic-fixture drift. The
sweep is gated by `*_TRACKING_DIR` env vars; it skips with explicit
reason when unset (memory: silently-skipping-tests-hide-breakage).

### 9. Tracking-aware features explicitly DEFERRED

PR-1 ships primitives only. The deferred priority order, per the
2026-04-30 lakehouse cross-check session:

1. `action_context()` — tracking-aware VAEP/xG features.
2. `pressure_on_carrier()`.
3. `infer_ball_carrier()` — heuristic per-frame carrier inference
   (lakehouse session addition).
4. `sync_score()` — per-action tracking↔events sync-quality score
   (lakehouse session addition).
5. Pitch-control models (Spearman / Voronoi).
6. Smoothing primitives (Savitzky-Golay, EMA).
7. Multi-frame interpolation / gap filling.
8. ReSpo.Vision adapter (licensing-gated, not engineering-gated).

PR-2 (`action_context`) targets silly-kicks 2.8.0 against the locked
2.7.0 schema.

## Consequences

- **Positive:** Future PR-2+ builds on locked schema + linkage primitives
  without re-ratifying. Lakehouse boundary stays one-direction-of-
  conversion (long-form → wide-form, 105×68 → 120×80, object id → bigint
  id), no library changes needed. Adding a new tracking provider
  reduces to: new adapter module + new fixture generator + new probe
  entry. Schema is invariant.
- **Negative:** A 5th provider whose source schema cannot fit the 19-column
  shape would force a schema revision; the four chosen (PFF / Sportec /
  Metrica / SkillCorner) cover the prevailing heterogeneity (JSON / JSONL
  / CSV / XML; 10/25/30 Hz), so the risk is low for the foreseeable
  expansion path.
- **Bundled refactor:** The `_direction.py` helper is extracted from
  `silly_kicks/spadl/pff.py` (PR-S18) into
  `silly_kicks/tracking/_direction.py` so events PFF, tracking PFF, and
  tracking Sportec adapters share one implementation. Pure refactor —
  zero behaviour change in events.

## References

- `docs/superpowers/specs/2026-04-30-tracking-namespace-pr1-design.md` — full PR-1 design.
- `docs/superpowers/plans/2026-04-30-tracking-namespace-pr1.md` — implementation plan.
- ADR-001 — converter identifier conventions (cross-namespace consistency).
- ADR-003 — NaN-safety contract for enrichment helpers (sets the precedent for
  marker-decorator + auto-discovered fuzz used elsewhere in the project).
- Bekkers, J. (2024). *DEFCON-style pressure metrics from tracking data*.
- Decroos, T. & Davis, J. (2020). *Player Vectors* / VAEP with tracking-aware features.
- Lakehouse cross-check session (2026-04-30): `karstenskyt__luxury-lakehouse`
  `fct_tracking_frames` mart — 38M rows, 3 providers, 20 matches.
