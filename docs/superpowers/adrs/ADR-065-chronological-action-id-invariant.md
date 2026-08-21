# ADR-065: Chronological `action_id` is an order-insensitivity invariant, enforced at the converter choke point

| Field | Value |
|---|---|
| **Date** | 2026-08-21 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

Several SPADL converters assigned `action_id = range(len(events))` over the **raw event array order** with no chronological sort (`sportec.py:656`, `gradientsports.py:764`), and derived shift-based fields (`_derive_end_coordinates`, `_add_dribbles`, provider result/type logic) positionally over that same unsorted order. The `action_id`-is-chronological property was an **assumed, inconsistently-enforced** invariant: `kloppy`/`opta`/`statsbomb` sort by time before assigning `action_id`; `skillcorner` sorts non-stably; `sportec`/`gradientsports`/`metrica`/`wyscout` did not. Consumers depend on it — `add_restart_coordinates`, VAEP labels, `secured_reception`/`_resolve_next_touch_positions`, `retains()`, and `_gk_geometry` (documented at `_gk_geometry.py:412`) all read `(game_id, period_id, action_id)` as chronological; VAEP labels and `secured_reception` **raise** `"time_seconds must be non-decreasing"` when it is not.

The forcing function: the lakehouse full-adoption effort reported non-chronological `action_id` from the sportec + gradientsports converters. Confirmed on committed data — `idsse_events_native_golden.parquet` carries a raw-order `28.806 → 26.370` inversion.

A permutation-invariance gate (convert an input, convert a timestamp-block permutation of it, compare outputs as content-multisets) **discovered the true broken set is larger than a source read**: `sportec`, `gradientsports`, `metrica`, `wyscout`, `skillcorner`, `opta` are all order-dependent — vindicating gate-driven over audit-driven scoping.

## Decision

A SPADL converter is a **pure function of its chronological event content**: permuting the input row order must yield identical output modulo the `action_id` renumber. Each order-dependent converter sorts at the TOP of its frame via the shared `spadl.base.sort_actions_chronologically(frame, *, by, tiebreak)` before any positional/`.shift()` derivation. The invariant is enforced two ways: a **permutation-invariance gate** over every `convert_to_actions` (red-first, CI), and a **raise-by-default runtime guard** `_assert_chronological_action_id` at the `_finalize_output` choke point (all converters + `convert_to_atomic` pass through it). Consumers that read persisted marts and do neighbour lookups adopt the robust `(game_id, period_id, time_seconds, action_id)` sort key, not `action_id` alone.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Output index-chronology check only (`action_id`-order == time-order) | Cheap, direct | Cannot see shift-derived corruption (skillcorner sorts before `action_id`, so its index IS chronological while `same_team_next`/`is_short` are still order-corrupt) | Necessary but not sufficient; kept only as a complementary gate |
| B. Fix only the source-read set (sportec + GS) | Small diff | Audit-driven; a source read missed `skillcorner`/`wyscout`/`metrica`/`opta` (the gate found all six) | Would leave four converters order-dependent |
| C. Order-insensitivity invariant + gate-driven fix + `_finalize_output` raise-guard (chosen) | Complete by the gate, not by a hand-list; fails fast at the boundary | A repo-wide contract + a retrain trigger for the genuinely-non-chronological providers | — |

## Consequences

### Positive
- Every converter is order-insensitive; a permuted or re-persisted feed produces identical output.
- The raise-guard turns a class of silent downstream corruption (wrong-neighbour geometry, mis-ordered VAEP labels) into a loud failure at the converter boundary.
- `_gk_geometry`/`add_restart_coordinates` and the other mart-reading consumers are robust to a legacy mart's non-chronological `action_id`.

### Negative
- **Retrain trigger** for providers whose REAL feed is non-chronological — **measured per-provider, not inferred from the gate** (the gate proves order-*dependence*; retrain is real-data output *change*): `sportec`/`IDSSE` (real inversion) and `wyscout` (a chronology bug fix — post-`:277`-sort row inserts scrambled output regardless of input order) genuinely change. **Measured byte-identical on real data → NOT retrain triggers:** `gradientsports` (all 64 real WC2022 matches) and `skillcorner` (8 real public matches — a chronological input pre-sort leaves the output unchanged). `opta` is fixture-verified + f24-time-ordered (not in the pining corpus). `metrica` is **not in the pining corpus**, so its real-data retrain status is unmeasured (fixture-verified only; a real-data M-C is recommended before lakehouse re-materialization).
- **Breaking input-contract change for Gradient Sports:** `EXPECTED_INPUT_COLUMNS` gains a required `start_time` (raw absolute clock), because a null-`startGameClock` FOUL (28/28 fouls across 13/64 matches) has no game clock and must have its `time_seconds` imputed by a **`start_time`-ordered** ffill to be order-insensitive. `gameEventId` was measured unfit (~21.5% non-chronological vs the game clock); `start_time`/`eventTime` have 0/144,374 inversions and are present on every foul. The lakehouse GS shaper must supply `start_time` (`event_time` fallback) or GS conversion raises the missing-column error.
- **`action_id` is a cross-table join key → ATOMIC migration.** Consumers must re-convert all affected-provider data together (bronze, goldens, id-keyed derived tables); a partial re-conversion misjoins silently.

### Neutral
- The permutation gate proves **inter-timestamp** order-insensitivity only (it preserves within-timestamp order); the correctness of `.shift()` *between co-timestamped events* is trusted to each converter's `tiebreak` (GS uses `start_time`; opta/skillcorner/wyscout rely on native intra-timestamp order via mergesort stability).
- No atomic-SPADL converter change (it derives from corrected standard SPADL); the raise-guard covers it via `convert_to_atomic`'s `_finalize_output` call.

## CLAUDE.md Amendment

Adds a durable converter-conventions contract to "Key conventions": converters are order-insensitive; sort at the top via `sort_actions_chronologically`; the invariant is CI-gated by the permutation gate and enforced at runtime by the `_finalize_output` raise-guard; mart-reading consumers sort `(…, time_seconds, action_id)`; the Gradient Sports converter requires a `start_time` input and imputes null-clock FOUL `time_seconds` by a `start_time`-ordered ffill.

## Related
- **Specs:** `docs/superpowers/specs/2026-08-20-chronological-action-id-invariant-design.md`
- **Plans:** `docs/superpowers/plans/2026-08-20-chronological-action-id-invariant.md`
- **Issues / PRs:** PR-S159

## Notes

Measured on the real WC2022 Gradient Sports corpus (`scratchpad/gs_foul_id_probe.py`, `gs_mc_realdata.py`; 64 matches, 144,541 events, 28 null-clock FOULs): native array order 0/144,374 inversions vs `startGameClock`; `gameEventId` 31,090/144,374 (~21.5%); `start_time`/`eventTime` 0/144,374 and present on all 28 fouls; OLD-vs-NEW converter output byte-identical (`foul_time_diff=0`, `order_diff=0`).
