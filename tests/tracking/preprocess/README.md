# TF-65 C1 — golden/snapshot blast-radius audit (plan Task 10b)

C1 reworks `smooth_frames` / `derive_velocities` (dense-grid reindex, `game_id` key, gap-adjacent
edge-NaN, new accel + uncertainty columns, soft guard). This note records, per committed parity oracle /
snapshot in the default (`not e2e`) suite, whether C1 moves it — so "full suite green" at the C1 boundary
cannot silently rewrite a committed baseline.

**Verdict: every committed oracle is byte-identical under C1** (verified: full CI-faithful suite on x86/CI +
an independent scratch-clone run — 91 golden/snapshot tests green; the only non-green anywhere is the
aarch64 last-ULP float note below, which is a platform artifact, not a C1 move).

| Oracle | Verdict | Why byte-identical |
|---|---|---|
| `tests/tracking/_fixtures/das_golden/` (`reference_das_*.csv`, `scenes_frames.csv`) via `_das_golden.py` + `test_das_golden_fixture.py` | byte-identical (x86/CI) | `scenes_frames.csv` carries `vx`/`vy` as FIXTURE INPUTS; `_das_pack.py` reads them directly; `_generate.py` imports zero `silly_kicks` and never calls the preprocess path. **NOT regenerated.** NOTE: on aarch64 (DGX) `test_golden_regenerates_byte_for_byte` fails by a last-ULP float (`…735` vs `…733`) — a platform artifact of ARM vs the x86-committed golden, independent of C1. |
| player-influence / pressure / empirical-AC snapshots | byte-identical | C1 is byte-identical on contiguous single-detection-run groups (reindex no-op, `game_id` no-op on single-game, guard no-op, edge-NaN only at gap-adjacent boundaries). These fixtures are contiguous. |
| `tests/tracking/_fixtures/gk_geometry_golden_frames.parquet` | byte-identical | TF-64 territory (keeper geometry); C1 velocity path does not move it. |
| reflection / mirror-invariance / metric_contracts / glossary gates | green | derived kinematic columns follow the `vx`/`vy` convention (declared only in `reflection.py`, not `TRACKING_FRAMES_COLUMNS`/metric_contracts/glossary); `accel_x`/`accel_y` enumerated in `_reproject_rows`. |

No oracle required an owner-signed-off regeneration.
