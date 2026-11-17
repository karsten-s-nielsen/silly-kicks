# IDSSE (Sportec/DFL) production-shape fixture

> **How the tests read these files (R66).** `sample_match.parquet` and `per_period_match.parquet` are **bronze**
> (the DFL parser's output). The tests read them only through `tests/_idsse_fixtures.py`, which takes them through
> `shape_events_to_native`, the converter's documented input (it resolves the team and player of the set-piece and
> foul rows from the qualifiers); `tests/test_idsse_fixture_path.py` pins that no other test file reads them. The
> documented `home_team_start_left` of each file below is unchanged.

`sample_match.parquet` is a subset of one DFL DataHub match (`idsse_J03WMX`)
sourced from luxury-lakehouse production `soccer_analytics.bronze.idsse_events`,
retained at the bronze schema level (no aggregation, no enrichment).

## Provenance

- Source: the public match J03WMX of Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). "An integrated dataset of spatiotemporal and event data in elite soccer". Scientific Data 12, 195. doi:10.1038/s41597-025-04505-y. Data set: figshare, doi:10.6084/m9.figshare.28196177, CC BY 4.0. The file is 308 events of that match (a subset), parsed by
  the luxury-lakehouse ingestion pipeline into `soccer_analytics.bronze.idsse_events` and taken from that bronze on
  2026-04-29. It is not built again (R65): against this library's parse of the public file its cells are equal,
  except `_ingested_at` (a lakehouse column) and `period` / `timestamp_seconds` on 6 `BallClaiming` rows, which hold
  the old parser's period and time and change no action.
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

CC BY 4.0, the licence of the published data set (Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). "An integrated dataset of spatiotemporal and event data in elite soccer". Scientific Data 12, 195. doi:10.1038/s41597-025-04505-y; figshare
doi:10.6084/m9.figshare.28196177). This fixture is included **only for testing** the silly-kicks open-source
library. Test fixtures are excluded from the published `silly-kicks` wheel
via `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]` in
`pyproject.toml`.

## Regenerating

```bash
# Requires DATABRICKS_HOST / DATABRICKS_TOKEN / DATABRICKS_HTTP_PATH env vars
uv run python scripts/extract_provider_fixtures.py --provider idsse
```

---

## `per_period_match.parquet` (PR-S23 / silly-kicks 3.0.1; rebuilt R64)

Full match J03WMX at the bronze schema level, exactly as THIS library's DFL parser gives it. It exists to exercise
the per-(team, period) direction-of-play invariant
(`tests/invariants/test_direction_of_play.py::test_per_team_per_period_shots_attack_high_x`) that the 2-shot
`sample_match.parquet` fixture cannot physically support, and the idsse penalty row (R57/R63).

### Provenance (R64, bar A26)

- Built by `scripts/build_idsse_per_period_fixture.py` from the two raw public DFL files of match `DFL-MAT-J03WMX`
  through `parse_dfl_match_info` + `parse_dfl_events`. No Databricks and no lakehouse table is read, and no column
  is added or changed after the parse: `match_id` is the parser's form, the bare DFL id `J03WMX`.
- Public source: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). "An integrated dataset of spatiotemporal and
  event data in elite soccer". Scientific Data 12, 195. doi:10.1038/s41597-025-04505-y. Data set: figshare,
  doi:10.6084/m9.figshare.28196177 (version 1, 2025-02-02), CC BY 4.0. See `NOTICE`.
- Raw files (public name; size; md5 as on the figshare record; sha256):
  - `DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml`; 11,930 bytes;
    md5 `bc5d5b21dbcf36aa30f4cd0428309f6b`; sha256 `a884728647a564cf8b20e5223a1d04a5da6f6725e1e506c8651505327c197ba9`
  - `DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml`; 776,578 bytes;
    md5 `2d88e180a1f63c7e4157feb980fc7255`; sha256 `6f9741ea62e9dee35b604942e1dad2ac0a4dc3fce14ef2f3858aa12b74dafe00`
- How the build got them: `scripts/_loader_pining.py` (`_download`) fetched them from the pining data service
  (`GET {PINING_API_URL}/idsse/matches/DFL-MAT-J03WMX/{artifact}`, default base
  `https://ozqgk9a3ji.execute-api.us-east-1.amazonaws.com/v1`) into a corner-build cache on 2026-11-17, stored as
  `idsse_DFL-MAT-J03WMX_metadata.xml` and `idsse_DFL-MAT-J03WMX_events.xml`. Their size and md5 equal the public
  figshare files above, so the files from figshare give the same build.
- Built file (sha256): `b6360284669b30b06ccbee0c89aaba222ee5ace566d5e2fe1cc56700e8e5cb55`; 1,715 events, 246 columns.
- Before R64 this file was the luxury-lakehouse production bronze, pulled through Databricks by
  `scripts/extract_provider_fixtures.py` (2026-04-29). That source is retired for this file (the script refuses its
  idsse `per_period` path). The rebuild differs from the old file in: `match_id` (`J03WMX`, was `idsse_J03WMX`) on
  every row; the lakehouse column `_ingested_at` (gone); the row order; 25 columns `string` where they were `object`;
  27 events in period 1 with positive times where the old file had them in period 2 with negative times (the
  parser's period hardening); and the 5 outcome cells of the scored penalty 18226500001426 (R63).
- Per-period orientation signature (raw, pre-conversion): home attacks LEFT in P1 -> `home_team_start_left=False`.

### Regenerating

```bash
python scripts/build_idsse_per_period_fixture.py \
    --match-info <raw>/DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml \
    --events <raw>/DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml
```

The output must match the sha256 above for the raw files above.

---

## `paired_tracking.parquet` (PR-S42 / silly-kicks 3.15.0)

Tracking frames from match `J03WMX` covering two time windows aligned with
events in `per_period_match.parquet`. Enables paired events+tracking testing
for Bug #2 (GK fallback) and Bug #7 (end-coordinate derivation).

### Provenance

- Source: the same public data set as `per_period_match.parquet` (Bassek et al. 2025; figshare
  doi:10.6084/m9.figshare.28196177), pulled from `soccer_analytics.dev_gold.fct_tracking_frames` via Databricks.
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

CC BY 4.0, the licence of the published data set, as `per_period_match.parquet` (the figshare record has one
licence for all its files).
Test-only fixture excluded from the published wheel.
