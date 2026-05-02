# Metrica production-shape fixture

`sample_match.parquet` is a 300-row subset of Metrica Sample Game 2,
converted to the bronze shape expected by `silly_kicks.spadl.metrica`.

## Provenance

- Source: [`metrica-sports/sample-data`](https://github.com/metrica-sports/sample-data)
  Sample Game 2, vendored upstream into kloppy's test corpus and then into
  silly-kicks at `tests/datasets/kloppy/metrica_events.json`.
- Subset: first 300 events with deterministic ordering, converted to
  bronze schema (`match_id`, `event_id`, `type`, `subtype`, `period`,
  `start_time_s`, `end_time_s`, `player`, `team`, `start_x`, `start_y`,
  `end_x`, `end_y`).
- Type counts: PASS 81, CARRY 110, BALL LOST 34, RECOVERY 32, CHALLENGE 20,
  SET PIECE 13, BALL OUT 8, FAULT RECEIVED 1, SHOT 1.
- No native GK markers exist anywhere in the Metrica event taxonomy —
  this is exactly the empirical situation that PR-S10 Bug 3 fixes by
  introducing the `goalkeeper_ids: set[str] | None` parameter on
  `silly_kicks.spadl.metrica.convert_to_actions`. Tests using this
  fixture supply the known goalkeeper player_ids from the source match.
- Extracted: 2026-04-29 via `scripts/extract_provider_fixtures.py --provider metrica`.
- File size: ~20 KB (bronze schema has 13 columns; small per-row size).

## License

Metrica Sample Game 2 is published under **CC-BY-NC-4.0**. This fixture is
included **only for testing** the silly-kicks open-source library
(non-commercial use). Test fixtures are excluded from the published
`silly-kicks` wheel via `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]`
in `pyproject.toml`.

## Regenerating

```bash
uv run python scripts/extract_provider_fixtures.py --provider metrica
```

(Reads `tests/datasets/kloppy/metrica_events.json` directly — no network.)

---

## `per_period_match.parquet` (PR-S23 / silly-kicks 3.0.1)

Full Metrica Sample Game 1 retained at the silly-kicks-input bronze
schema level. Exists specifically to exercise the per-(team, period)
direction-of-play invariant
(`tests/invariants/test_direction_of_play.py::test_per_team_per_period_shots_attack_high_x`)
that the period-1-only `sample_match.parquet` fixture cannot physically
support.

### Provenance

- Source: [`metrica-sports/sample-data`](https://github.com/metrica-sports/sample-data)
  Sample Game 1 (CC-BY-NC-4.0; same license as Sample Game 2 used in
  `sample_match.parquet`). Pulled via luxury-lakehouse SK3-MIG
  migration session.
- Coord system normalization: lakehouse bronze ships Metrica's native
  0-1 normalized coords; `extract_provider_fixtures.py --variant
  per_period` rescales to 0-105 / 0-68 (Metrica's standard 105 m × 68 m
  pitch) so the parquet schema matches `sample_match.parquet`.
- Row count: 1,745 events including 24 shots.
- Per-period orientation signature (post-rescale):
  - P1: 11 home shots near x ≈ 92, 2 away shots near x ≈ 17.
  - P2: 7 home shots near x ≈ 10, 4 away shots near x ≈ 86.
  - Each team alternates side between periods → PER_PERIOD_ABSOLUTE.
  - Home attacks RIGHT in P1 → `home_team_start_left=True`.
- File size: ~47 KB.

### Regenerating

```bash
uv run python scripts/extract_provider_fixtures.py --provider metrica --variant per_period
```
