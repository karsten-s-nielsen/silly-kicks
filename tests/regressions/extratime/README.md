# Real-data ET fixtures for PR-S70 Task 8

**Provenance:** extracted from luxury-lakehouse bronze on 2026-05-30 by the lakehouse maintainer in response to the spec §6/§8 + plan Task 8 ask.

## What's here

### `gs_et/` — Gradient Sports A-League ET match

- **Source:** lakehouse `bronze.gradientsports_tracking` + `bronze.spadl_actions` + `bronze.gradientsports_events`, `match_id = 10517`.
- **Window:** period 3 (first overtime), frames 197541..197999 (459 distinct frames).

| File | Rows | Notes |
|---|---|---|
| `frames.parquet` | 10,557 | 459 distinct frames × 22 players + 459 ball rows; columns: `match_id`, `period`, `frame_num`, `period_elapsed_time`, `team_side`, `is_ball`, `jersey_num`, `x`, `y`, `z` |
| `actions.parquet` | 1,838 | Full match across all 4 periods; **415 actions in periods 3/4 (ET)** |
| `meta.parquet` | 1 | `home_team_id=364`, `home_start_left=True` (RT), `home_team_start_left_extratime=True` (ET) — both flags verifiable against the converter output |

The meta carries the **true** ET start direction sourced from `stadiumMetadata.homeTeamStartLeftExtraTime`, so the converter tests can verify orientation correctness, not just no-raise.

### `sportec_idsse_et/` — NOT DELIVERED

**No IDSSE/Sportec ET match exists in lakehouse bronze.** The §8 historical audit (2026-05-30) found **zero** ET-period matches across both `bronze.spadl_actions` and `bronze.idsse_tracking`. The Bundesliga regular season has no ET; cup-competition data with ET hasn't been ingested.

Implications for the silly-kicks 4.0.0 PR:

1. **§8 historical-mis-orientation audit:** moot — no IDSSE/Metrica ET data exists in production, so the silent-default Sportec/Metrica path has produced zero mis-oriented data. The §8 remediation work is a no-op. silly-kicks can ship 4.0.0 to PyPI without waiting on lakehouse remediation.
2. **Task 8 IDSSE round-trip fixture:** synthesize one. A minimal Sportec-shaped DataFrame with `period_id IN (3, 4)` rows (mirroring the existing `tests/tracking/test_sportec_tracking*.py` input-builder shape) is sufficient for the per-converter unit tests. The cross-provider parity test (Task 6) and RT-only no-regress (Task 7) already cover Sportec at the unit level; the missing piece is just an end-to-end ET round-trip, which a synthetic fixture covers honestly.
3. When IDSSE cup-with-ET data is eventually ingested, the lakehouse will need a `derive_idsse_home_team_start_left_extratime` helper (analogous to the existing period-1 `derive_idsse_home_team_start_left`) before AC-1 can pass the ET flag to silly-kicks. Tracked separately as future Phase A.0 work.

### `metrica_et/` — NOT DELIVERED

Same as IDSSE: zero Metrica ET matches in lakehouse bronze. Synthesize if needed; same plan implications.

## Audit details

Full audit results memo: lakehouse `memory/project_et_direction_section_8_audit.md`. Summary:

| Provider | ET matches in bronze.spadl_actions | ET matches in bronze.tracking |
|---|---|---|
| IDSSE | 0 | 0 |
| Metrica | 0 | 0 |
| Gradient Sports | 5 | 3 |

## Validation hint

GS fixture quick-check:

```python
import pandas as pd
fr = pd.read_parquet("tests/regressions/extratime/gs_et/frames.parquet")
a  = pd.read_parquet("tests/regressions/extratime/gs_et/actions.parquet")
m  = pd.read_parquet("tests/regressions/extratime/gs_et/meta.parquet")
assert (fr["period"] == 3).all()                                            # all period-3 frames
assert fr["frame_num"].nunique() == 459 and len(fr) == 10557
assert a[a["period_id"].isin([3, 4])].shape[0] == 415                       # ET actions
assert m.iloc[0].to_dict() == {                                              # meta truth
    "home_team_id": 364, "home_start_left": True, "home_team_start_left_extratime": True
}
```
