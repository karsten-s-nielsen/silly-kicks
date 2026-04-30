# TODO

Open items tracked for future work. Closed items live in
[CHANGELOG.md](CHANGELOG.md) and git history.

## Tracking namespace — `silly_kicks.tracking.*`

Scope: bring tracking-data support into silly-kicks as a first-class library
surface, parallel to `silly_kicks.spadl.*`. Deferred from the PFF events-only
PR (silly-kicks 2.6.0) per scoping decision recorded in
`docs/superpowers/specs/2026-04-30-pff-fc-events-converter-design.md` § 6.

### Why tracking adds separate value

Events and tracking are complementary, not redundant. Events carry semantic
intent (cross vs long pass, set-piece type taxonomy, foul/card/VAR). Tracking
unlocks analytical surfaces events fundamentally cannot:

- Off-ball player positions: pressure measurement, defensive compactness,
  pass-option counts, off-ball runs, defender density between shooter and
  goal.
- Player kinematics: velocity, acceleration, sprint counts, high-intensity
  distance, fatigue proxies.
- Pre/post-event context windows (e.g., defender behaviour 3 s after a shot).
- Pitch-control / Voronoi / Spearman models — require every player's
  position at every frame.
- Tracking-aware VAEP: nearest-defender distance at action start,
  ball-carrier speed, receiver-zone density. Known to materially improve
  VAEP / xG / xPass calibration.
- Refined GK / pre-shot context — `add_pre_shot_gk_context` becomes
  genuinely accurate once GK position is known per-frame, not inferred.
- Forward-compatible QA layer: tracking can detect mistagged event
  locations, supporting silent-drift detection.

### Architectural rules (carried forward from PFF events-only design)

1. **Hexagonal, pure-function.**
   `silly_kicks.tracking.<provider>.convert_to_frames(raw_frames_df, roster_df, ...) -> tuple[pd.DataFrame, ConversionReport]`.
   Same shape as the SPADL converters.
2. **No I/O in the package.** Caller parses provider files
   (JSON, JSONL, CSV, XML+positions) into the provider-shaped input
   DataFrame. Caller-responsible, every consumer.
3. **Provider-agnostic at OUTPUT, provider-shaped at INPUT.** The
   `TRACKING_FRAMES_COLUMNS` schema is canonical; per-provider adapters
   translate native vocabularies into it.
4. **Library-native conventions, not lakehouse-specific.** Coordinates:
   silly-kicks SPADL 105 × 68 meters, NOT the lakehouse's StatsBomb 120 × 80
   units (which exists for legacy event-coordinate alignment). Identifier
   types: per-provider conventions matching the SPADL converter for the
   same provider (object for kloppy/sportec, int64/Int64 for PFF and
   StatsBomb-style providers), NOT synthetic
   `provider_match_side_jersey` FQNs.
5. **Long-form frames DataFrame.** One row per (frame, player). Ball is its
   own row (`is_ball=True`, `player_id=null`, `team_id=null`) — NOT a
   wide-form schema with per-player columns.
6. **One ADR at scoping time** (ADR-004 placeholder): "silly-kicks now
   supports tracking. Hexagonal pure-function contract. Canonical long-form
   frames schema. Per-provider adapter modules. No I/O in the package.
   Smoothing / possession windowing / pressure features deferred to future
   scoped releases."

### Canonical schema starter (informed by lakehouse evidence, not copied from it)

The luxury-lakehouse `fct_tracking_frames` mart already runs production
tracking across 3 providers (Metrica, IDSSE/Sportec, SkillCorner) on 20
matches / ~38M player-frame rows (verified 2026-04-30 via Databricks probe
of `soccer_analytics.dev_gold.fct_tracking_frames`). Its 15-column shared
schema is strong prior art for which fields a tracking schema needs.
silly-kicks should adopt the **field set**, not the dtype/unit choices
(see rule 4 above). Starter draft, subject to revision at scoping:

```python
TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    "game_id":         "int64 | object",  # provider-dependent, mirrors SPADL
    "period_id":       "int64",
    "frame_id":        "int64",            # frame within period
    "time_seconds":    "float64",          # period-relative, like SPADL
    "frame_rate":      "float64",          # Hz (10 / 25 / 30 fps real-world)
    "player_id":       "int64 | object",   # null for ball rows
    "team_id":         "int64 | object",   # null for ball rows
    "is_ball":         "bool",
    "is_goalkeeper":   "bool",
    "x":               "float64",          # SPADL meters [0, 105]
    "y":               "float64",          # SPADL meters [0, 68]
    "z":               "float64",          # null for non-ball rows that lack z
    "speed":           "float64",          # m/s; null when provider doesn't supply
    "confidence":      "object",           # provider-specific tier
    "visibility":      "object",
    "source_provider": "object",
}
```

### Minimum viable surface (Y-lite)

- `silly_kicks.tracking.schema` — `TRACKING_FRAMES_COLUMNS`,
  `TRACKING_CONSTRAINTS`. Reuses existing `ConversionReport`.
- `silly_kicks.tracking.<provider>` for ≥3 of {pff, sportec, metrica,
  skillcorner} — one canonical adapter per provider,
  `convert_to_frames(...)`.
- `silly_kicks.tracking.utils` — exactly one cross-pipeline utility on day
  one: `link_actions_to_frames(actions, frames, tolerance_seconds=...)`.
  This is the single operation downstream consumers cannot easily reproduce
  themselves and that everything else (pressure features, possession
  windows, tracking-aware VAEP) eventually requires. Plus
  `play_left_to_right` analog for tracking (mirror of SPADL's existing
  helper).
- One ADR (ADR-004).

### Provider candidates (open data validated as of 2026-04-30)

Verified by inspecting `karstenskyt__luxury-lakehouse` and a read-only
Databricks probe of `soccer_analytics.dev_gold.fct_tracking_frames`:

| Provider          | Open data                                            | Format            | Hz  | Lakehouse evidence (2026-04-30) |
|-------------------|------------------------------------------------------|-------------------|-----|----------------------------------|
| Metrica           | 3 sample games, GitHub `metrica-sports/sample-data`  | CSV               | 25  | 3 matches, 9.5M player-frames    |
| IDSSE / Sportec   | Bassek 2025 *Bundesliga Match Data Open Dataset*     | XML+positions     | 25  | 7 matches, 21.9M player-frames   |
| SkillCorner       | ~9 open broadcast-tracking matches on GitHub         | JSON              | 10  | 10 matches, 6.8M player-frames   |
| PFF FC            | WC 2022 (signup-gated)                               | JSONL.bz2         | 30  | not loaded; primary candidate    |
| TRACAB / SecondSpectrum / Hawkeye / InStat | none public                          | —                 | —   | excluded                          |
| ReSpo.Vision      | partnership-only releases reported                   | broadcast-derived | —   | future contact; flag for licensing inquiry once an open subset becomes available |

Schema-stress test goal: lock the canonical schema only after 4 providers
load cleanly into it (multi-format CSV/JSON/JSONL/XML, multi-Hz 10/25/30,
broadcast-derived + optical + native — heterogeneity prevents PFF-shaped
lock-in).

### Decision points to revisit at scoping time

1. **Native adapter per provider vs kloppy gateway + 1–2 natives.** Kloppy
   already parses tracking from SkillCorner, Sportec, Metrica, TRACAB, and
   SecondSpectrum — a `silly_kicks.tracking.kloppy` gateway (mirroring the
   existing `silly_kicks.spadl.kloppy`) would reach all of them with one
   adapter. Trade-off: depth of native adapters vs breadth of kloppy reach.
   PFF lacks a kloppy parser — must be native regardless.
2. **Long-form schema ball-row encoding.** Confirm `is_ball=True` separate
   row beats wide-form alternatives (e.g., per-frame `ball_x` / `ball_y`
   columns broadcast across player rows, as the lakehouse does). Both work;
   pick after writing the linkage utility against both shapes.
3. **Speed / acceleration computation.** Some providers supply `speed`
   natively (PFF, IDSSE); others don't (Metrica, SkillCorner). Decide:
   trust source where present, derive elsewhere via lag/diff (lakehouse
   pattern), or always derive for consistency. The lakehouse uses
   per-row `frame_rate` to support mixed-Hz unions cleanly — same trick
   applies here.
4. **Test-fixture policy.** Same constraint as the events converter: tests
   cannot depend on local data not in the repo. Synthetic tracking
   fixtures (one per supported provider) become load-bearing. Generator
   scripts in `tests/datasets/<provider>/` mirror the events pattern.
5. **Coordinate-system reaffirmation.** The events converter already runs
   in 105×68 meters; the tracking adapter must produce the same. ADR-004
   explicitly forbids re-importing the lakehouse's StatsBomb-units choice.
6. **Direction-of-play normalization** has the same per-period flip issue
   as PFF events; the same `home_team_start_left[_extratime]` parameters
   apply. Reuse the helper carved out for the events converter.

### Deferred (explicitly NOT in the eventual tracking-namespace PR)

- Smoothing primitives (Savitzky-Golay, exponential moving average).
- Possession-window slicing (`slice_around_event`, etc.).
- Pressure / defender-density features.
- Tracking-aware VAEP / xPass features.
- Pitch-control models (Spearman, Voronoi).
- Multi-frame interpolation / gap filling.
- Streaming / chunked reading for large tracking files.
- ReSpo.Vision adapter (pending licensing).

Each of these gets its own scoping cycle once the canonical schema and
linkage utility are stable and a concrete consumer use case is established.

### Validation plan

- All 4 candidate tracking providers (sportec, metrica, skillcorner, pff)
  must load tracking data cleanly into `TRACKING_FRAMES_COLUMNS` before
  the schema is locked. Note: skillcorner has tracking only (no events
  in silly-kicks today); sportec, metrica, and pff have both.
- `link_actions_to_frames` round-trip test: events ↔ frames join must
  produce non-empty results on every provider's synthetic-match pair.
- `play_left_to_right` for tracking must produce the same orientation
  invariants as SPADL `play_left_to_right`.
- Cross-provider parity test: distance-to-ball distributions per
  source_provider must overlap (proves coordinate normalization is
  consistent across providers).

## Documentation

(none currently queued — D-8 closed in silly-kicks 2.1.1)

## Architecture

(none currently queued — A9 closed in silly-kicks 2.4.0 via `silly_kicks.vaep.feature_framework` extraction; see ADR-002)

## Open PRs

(none currently queued — PR-S14 shipped in silly-kicks 2.2.0)

## Tech Debt

(none currently queued — A19 / O-M1 / O-M6 reviewed and closed in 2.3.0 as stale-or-by-design; D-9 closed in 2.1.1; C-1 closed in 2.2.0)
