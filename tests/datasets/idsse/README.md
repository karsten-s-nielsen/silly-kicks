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
