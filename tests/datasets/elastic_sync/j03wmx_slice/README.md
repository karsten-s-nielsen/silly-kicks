# TF-57 ELASTIC-NW sync — CC-BY oracle slice (J03WMX)

A reduced, committed slice of the **DFL-MAT-J03WMX** match used as the in-repo regression
oracle for the extended-Needleman–Wunsch event↔tracking sync (`tracking/_elastic_sync.py`,
TF-57). The CI gate (`tests/tracking/test_elastic_sync_oracle.py`) scores
`align_events_to_frames` against Kim et al.'s frame-level ground truth and asserts a
regression floor + a large-margin beat over the retired greedy aligner.

## Licence (CC BY 4.0) and attribution — REQUIRED

All three parquets are **CC BY 4.0**. They derive from:

- **Sportec Open DFL Dataset** — Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025).
  "An integrated dataset of spatiotemporal and event data in elite soccer." *Scientific Data*.
  doi:10.1038/s41597-025-04505-y. (Base tracking + events; CC BY 4.0.)
- **ELASTIC benchmark re-annotation** — Kim, H., Choi, H., Lee, K., Seo, S., Boomstra, T.,
  Yoon, J., & Park, C. (2026). "ELASTIC: Trajectory-Based Synchronization of Event and Tracking
  Data in Soccer." CIKM 2026, arXiv:2608.30227.
  Repo: github.com/hyunsungkim-ds/elastic (branch `cikm2026`). The `benchmark/` ground-truth
  (`frame_id`, `receive_frame_id`) is **redistributed by the ELASTIC repo under CC BY 4.0** —
  README verbatim: *"The event data under `benchmark/` is derived from the Sportec Open DFL
  Dataset (Bassek et al., 2025)… We redistribute it under the same license, with the
  modifications described above."* (Grant verified 2026-09-11.)

**No MPL-2.0 code or data is included.** ELASTIC's source (MPL-2.0) is never vendored; the
sync-accuracy metric in the CI gate is reimplemented from scratch (`|pred − gt| ≤ N`).

Any redistribution of these files must carry the two citations above.

## Contents

| file | rows | what |
|---|---|---|
| `frames.parquet` | 142 623 | tracking frames, `TRACKING_FRAMES_COLUMNS` schema (period 1, frame_id 19950–26150; ~240 s at 25 fps; ball + players; DFL PersonId/CLU ids; `z` present but sparse — broadcast tracking) |
| `actions.parquet` | 67 | sync-input SPADL-shaped actions, row-aligned to `gt` by `action_id` (see below) |
| `gt.parquet` | 67 | Kim's frame-level ground truth: `action_id, frame_id` (event START), `receive_frame_id` (event END / reception), `receiver_id`, `spadl_type`, `period_id` |

Span: **period 1, [400 s, 640 s]** (chosen to COVER a goal — the period-1 goal at t≈468 s,
frame 21706 — while retaining all four ELASTIC categories). Category coverage: open-play 51
(pass 50 / shot 1), minor 10 (bad_touch 5 / tackle 5), set-piece 1 (goalkick 1), incoming 5
(interception 5). **51 receptions, 1 out, 1 goal.** All 19 acting players are present in `frames`
(the actor-membership gate is fully live); every `gt.frame_id` and `receive_frame_id` lies inside
the `frames` window.

## Reduction recipe (reproducible; script stays off-repo)

Produced on the DGX by `~/elastic_validation/make_oracle_slice.py` (off-repo — it reads the
uncommitted tc3-cache + the CC-BY bench gt):

```
python make_oracle_slice.py --period 1 --t0 400 --t1 640 --out <dir>
```

Method:
1. Kim's `unsynced[i]` and `gt[i]` are the **same event, row-aligned** → `action_id := row index`.
2. Kim's provider UTC is anchored to our per-period frame clock via `gt.frame_id`
   (`kickoff_p = median(utc − (gt.frame_id − frame_base_p)/25)`), so `actions.time_seconds`
   carries the **real provider misalignment** the sync must correct.
3. Kim's synthetic `home_N`/`away_N` player labels do not match our DFL PersonIds, so each label
   is mapped to a real `DFL-OBJ-…` id + `DFL-CLU-…` team via a **match-wide majority vote** over
   our SPADL actions matched by `(type, anchored-time, location)`. Span coverage: **67/67 (100 %)**.
4. Kim's benchmark taxonomy → SPADL `type_name` (three Kim-only tokens mapped to the nearest
   SPADL touch type):

   | Kim `spadl_type` | SPADL | note |
   |---|---|---|
   | pass, cross, shot, shot_penalty, clearance, throw_in, goalkick, freekick_short/crossed, corner_short/crossed, interception, tackle, foul, bad_touch, keeper_save/claim/pick_up | (identical) | direct |
   | `dispossessed` | `bad_touch` | lost ball under pressure → minor miscontrol |
   | `ball_recovery` | `interception` | regained loose ball → incoming |
   | `shot_block` | `clearance` | defender blocked the shot → defensive outgoing touch |

## Honest limitations (load-bearing)

- **`player_id`/`team_id` are inferred** (match-wide vote map), not native to Kim's gt — a
  faithful stand-in for the actor-membership gate, not ground-truth identity. A mis-mapped actor
  can only *lower* measured accuracy (a wrong actor fails the membership gate), so the oracle is a
  conservative floor, never an inflated one.
- **Goal coverage.** This span COVERS the period-1 goal (t≈468 s, frame 21706), so the accuracy
  oracle exercises the **goal virtual-event insertion rule** directly. Spec §8.1 requires all four
  categories plus ≥1 of *out / **goal** / reception*; this slice carries 51 receptions, 1 out, and
  1 goal — so it meets the requirement in full (the earlier `[720,960]` span carried no goal). The
  goal path is additionally unit-tested by `tests/tracking/test_elastic_sync.py::TestEventEnrichment`.
- This is a **regression floor** on one committed slice, not the headline gap-closure — that is
  the full-corpus DGX report (spec §8.2), run on all three CC-BY matches.
