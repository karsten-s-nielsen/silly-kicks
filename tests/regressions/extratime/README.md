# Real-data ET fixtures for PR-S70 Task 8

**Provenance:** extracted from luxury-lakehouse bronze on 2026-05-30 by the lakehouse maintainer in response to the spec §6/§8 + plan Task 8 ask.

## What's here

### `gs_et/` — Gradient Sports **WC2022 knockout** ET match (native GK)

- **Source:** Gradient Sports (PFF FC) WC2022 `match_id = 10517` — a knockout fixture with extra time.
  (The pre-2026-06-19 version of this file labelled it "A-League"; that was a maintainer typo —
  GS in this repo is WC2022, A-League tracking is SkillCorner.)
- **Window:** period 3 (first overtime), the first 500 distinct frames.
- **Regenerated 2026-06-19** by `scripts/regenerate_gs_et_native_gk.py` (from the pining cache) as the
  **raw `tracking.gradientsports.convert_to_frames` INPUT** carrying the **native** `is_goalkeeper`
  from the roster join — so `test_real_et_roundtrip.py` exercises the production GK anchor (TF-23b,
  ADR-035). No extra restricted fields beyond the adapter input contract are committed.

| File | Rows | Notes |
|---|---|---|
| `frames.parquet` | 11,500 | 500 frames × 22 players + 500 ball rows; the GS tracking-adapter `EXPECTED_INPUT_COLUMNS` (`game_id`, `period_id`, `frame_id`, `time_seconds`, `frame_rate`, `player_id`, `team_id`, `is_ball`, `is_goalkeeper` [native], `x_centered`, `y_centered`, `z`, `speed_native`, `ball_state`) |
| `meta.parquet` | 1 | `home_team_id=364`, `away_team_id=363`, `home_start_left=True`, `home_team_start_left_extratime=True` |

**Do NOT treat `home_team_start_left_extratime` as orientation ground truth.** It is the constant
GS-ET `stadiumMetadata` placeholder this feature (TF-23b) exists to correct, and for 10517 P3 it is
geometrically **wrong** (it leaves the home GK on the attacking half — the old bounds-only test
passed anyway because both orientations are in-bounds). `test_real_et_roundtrip.py` instead asserts
the **geometric** truth (a defending GK sits deep in its own half → home GK on the LOW-x half in the
home-attacks-right frame) and that the backstop reaches it under both the flag and its negation.

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

The Gradient Sports row (5 ET in events / **3 in tracking**) bounds the TF-23b retrain scope: the
backstop is **tracking-only**, so only the ≤3 ET-tracking matches with a wrong placeholder flag can
change (events converters are untouched). The exact changed set is the ADR-035 G1 non-no-op list.

## Validation hint

GS fixture quick-check (raw adapter input, native GK):

```python
import pandas as pd
fr = pd.read_parquet("tests/regressions/extratime/gs_et/frames.parquet")
m = pd.read_parquet("tests/regressions/extratime/gs_et/meta.parquet").iloc[0]
assert (fr["period_id"] == 3).all() and fr["frame_id"].nunique() == 500 and len(fr) == 11500
# native home GK present (the production anchor the round-trip test exercises)
hg = fr[(fr["team_id"] == 364) & fr["is_goalkeeper"] & ~fr["is_ball"]]
assert hg["player_id"].nunique() >= 1
assert int(m["home_team_id"]) == 364
```
