# SB360 coverage — Layer B, real open data

**Question this answers:** Layer A established what silly-kicks' code *does* on freeze-frames.
This establishes what real freeze-frames *contain* — whether the frames exist at all for the
events a GK metric values, and whether the relevant keeper is in them.

## Provenance

| | |
|---|---|
| Driver | `scripts/build_sb360_coverage.py` |
| Commit | `650ed0749fff933a2bd69d76798ce5dcb6633380` |
| Tree | **clean** (`dirty: false`) |
| Generated | 2026-08-04 |
| Source | StatsBomb open data via `statsbombpy` |
| Pass | 22 matches attempted, 22 processed, 0 skipped, 0 failed |

| Cell | Competition / season | Matches used |
|---|---|---|
| 72/107 | Women's World Cup 2023 | 8 |
| 43/106 | FIFA World Cup 2022 | 8 |
| 44/107 | Major League Soccer 2023 | 3 (of 6 attempted — see exclusions) |

## Excluded and counted: 3 of 22 matches

`3877115`, `3877170`, `3877194` (all MLS 2023) ship a 360 file whose `event_uuid`s have **zero
overlap with their own events file**, while correctly claiming the same `match_id`. Verified
against the RAW events, not merely the SPADL actions, so this is an upstream inconsistency
rather than a converter or join defect.

They are excluded from every figure below and counted here rather than averaged over: each
would otherwise have contributed a single `unmapped` bucket, visually indistinguishable from a
quiet match, diluting every aggregate it entered.

**14% of sampled matches had unusable 360↔event linkage.** That is itself a planning fact.

## GK-domain coverage — 19 usable matches

Two independent quantities. **Frame existence** is counted from the ACTION side — how many
actions of that type occurred, and how many received a freeze-frame at all. Keeper visibility is
conditional on a frame existing, so it can only ever describe the covered subset.

Which keeper matters depends on the action: shots and crosses want the **defending** keeper;
distribution and saves want the **acting side's**, because there the keeper IS the actor. The
column that is definitionally zero for a type is marked "—".

| SPADL type | matches | actions | with frame | **frame existence** | defending GK | acting GK | mean players visible | visible pitch |
|---|---|---|---|---|---|---|---|---|
| `shot_penalty` | 6 | 6 | 6 | **100%** | 100% | — | 12.3 | 0.13 |
| `shot_freekick` | 11 | 19 | 19 | **100%** | 100% | — | 19.1 | 0.21 |
| `shot` | 19 | 473 | 462 | **97.7%** | **92.2%** | — | 14.7 | 0.18 |
| `keeper_save` | 19 | 111 | 108 | **97.3%** | — | **100%** | 12.8 | 0.16 |
| `cross` | 19 | 399 | 339 | **85.0%** | **81.4%** | — | 14.5 | 0.19 |
| `goalkick` | 16 | 258 | 84 | **32.6%** | — | **96.4%** | 9.5 | 0.30 |

## What this means

**Shot-facing and save GK analysis on SB360 is well covered.** Shots and saves carry a
freeze-frame ~98% of the time, and the relevant keeper is in it 92–100% of the time. Combined
with Layer A — where `add_pre_shot_gk_position`, `add_pre_shot_gk_angle`, `add_gk_completion`
and all 16 `add_xt_gk` columns produce identical values with or without velocity — the
shot-facing GK surface is usable on freeze-frames today.

**Goal-kick distribution is the constraint, and it is a frame-availability constraint.** Only
about a third of goal kicks carry a freeze-frame. When one exists the kicking keeper is in it
96% of the time, so the limit is not keeper visibility and not the library: it is whether
StatsBomb captured a frame for the event at all.

Dispersion matters more than the point estimate here. Per-match goal-kick frame existence over
16 matches: **median 21%, IQR 18–50%, range 8–61%**. A club cannot plan around 33% as though it
were a stable rate.

**Crosses sit in between** at 85% frame existence and 81% defending-keeper visibility.

## Reading limits

- **No causal cell contrast is claimed** (ADR-053; spec amendment). The unit of analysis is the
  MATCH — goal kicks within a match share one broadcast, one camera rig, one crew — so 8 matches
  is an effective n of ~8 clusters regardless of how many goal kicks they hold. A credible
  sex- or production-tier contrast needs ~15–20 matches per cell. The three cells are present
  for breadth of production conditions, not as a comparison.
- **MLS is under-represented** (3 usable matches) precisely because its exclusions fell there.
- `shot_penalty` (6) and `shot_freekick` (19) have small denominators; their 100% figures are
  consistent with the shot family but should not be quoted alone.
- **Open data is not delivered data.** These are the broadcasts StatsBomb happened to process
  for public release. An NWSL commercial feed may differ in either direction, and NWSL itself
  carries no 360 in the open data — which is the point of the collaboration.

## Reproducing

```bash
python scripts/build_sb360_coverage.py --matches-per-cell 8
```

Shards land in the gitignored top-level `sb360_coverage_shards/`, one per match, resumable. The
run refuses a dirty tree; `--allow-dirty` is available for development and stamps
`dirty: true` onto the artifact rather than hiding it.
