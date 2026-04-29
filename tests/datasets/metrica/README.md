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
