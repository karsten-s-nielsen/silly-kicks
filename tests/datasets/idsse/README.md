# IDSSE (Sportec/DFL) production-shape fixture

`sample_match.parquet` is a subset of one DFL DataHub match (`idsse_J03WMX`)
sourced from luxury-lakehouse production `soccer_analytics.bronze.idsse_events`,
retained at the bronze schema level (no aggregation, no enrichment).

## Provenance

- Source: DFL DataHub free-sample data, parsed by the luxury-lakehouse
  ingestion pipeline into `soccer_analytics.bronze.idsse_events`.
- Source match_id: `idsse_J03WMX` (Bundesliga; full match_id is a public
  DFL competition identifier — no PII).
- Subset rule: every Play event with non-null `play_goal_keeper_action` plus
  a stratified sample of other event types, capped at ~400 rows. Designed
  to exercise the throwOut + punt qualifier paths fixed in silly-kicks
  1.10.0 (PR-S10 Bug 2).
- Extracted: 2026-04-29 via `scripts/extract_provider_fixtures.py --provider idsse`.
- Row count: 308 (158 Play + 49 TacklingGame + 48 OtherBallAction + smaller
  type counts; 7 throwOut + 1 punt GK-distribution rows).
- File size: ~166 KB (DFL bronze has ~140 columns; per-row size is large
  even when most are NULL).

## License

DFL DataHub free-sample license permits non-commercial redistribution.
This fixture is included **only for testing** the silly-kicks open-source
library. Test fixtures are excluded from the published `silly-kicks` wheel
via `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]` in
`pyproject.toml`.

## Regenerating

```bash
# Requires DATABRICKS_HOST / DATABRICKS_TOKEN / DATABRICKS_HTTP_PATH env vars
uv run python scripts/extract_provider_fixtures.py --provider idsse
```

---

## `per_period_match.parquet` (PR-S23 / silly-kicks 3.0.1)

Full match `idsse_J03WMX` retained at the bronze schema level. Exists
specifically to exercise the per-(team, period) direction-of-play
invariant (`tests/invariants/test_direction_of_play.py::test_per_team_per_period_shots_attack_high_x`)
that the 2-shot `sample_match.parquet` fixture cannot physically support.

### Provenance

- Source: same DFL DataHub free-sample data as `sample_match.parquet`,
  pulled via the luxury-lakehouse SK3-MIG migration session.
- Match identifier: `idsse_J03WMX` (Bundesliga; public DFL competition
  identifier — no PII).
- Subset rule: full match (no row cap; events filtered for the
  silly-kicks bronze input shape only).
- Row count: 1,715 events including 20 shots.
- Per-period orientation signature (raw, pre-conversion):
  - P1: 5 home shots cluster near x ≈ 14, 4 away shots near x ≈ 92.
  - P2: 3 home shots near x ≈ 92, 8 away shots near x ≈ 18.
  - Each team alternates side between periods → PER_PERIOD_ABSOLUTE.
  - Home attacks LEFT in P1 → `home_team_start_left=False`.
- File size: ~250 KB (full bronze 247-column schema).
- Citation: see `NOTICE` "Test Data Sources" section.

### Regenerating

```bash
uv run python scripts/extract_provider_fixtures.py --provider idsse --variant per_period
```

---

## `paired_tracking.parquet` (PR-S42 / silly-kicks 3.15.0)

Tracking frames from match `J03WMX` covering two time windows aligned with
events in `per_period_match.parquet`. Enables paired events+tracking testing
for Bug #2 (GK fallback) and Bug #7 (end-coordinate derivation).

### Provenance

- Source: same DFL DataHub free-sample data, pulled from
  `soccer_analytics.dev_gold.fct_tracking_frames` via Databricks.
- Match identifier: `J03WMX` (Bundesliga; public DFL competition
  identifier).
- Time windows:
  - P1: 90.0 - 107.0s (4 events: 3 Play + 1 ShotAtGoal)
  - P2: 624.0 - 640.0s (9 events: 3 Play + 1 ThrowIn + 2 OtherBallAction + 1 TacklingGame + 1 ShotAtGoal)
- Row count: ~18,194 (22 players x ~827 frames).
- GK players: `DFL-OBJ-0002DR` (away), `DFL-OBJ-0002HE` (home).
- File size: ~518 KB.
- Extraction script: `scripts/extract_paired_idsse_fixture.py`.

### License

Same DFL DataHub free-sample license as `per_period_match.parquet`.
Test-only fixture excluded from the published wheel.
