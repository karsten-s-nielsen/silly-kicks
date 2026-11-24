# Tracking compute: the C1 before/after record (lakehouse calls)

This memo holds the before/after record of the tracking-batch cycle (spec
`docs/superpowers/specs/2026-11-08-tracking-batch-and-dfl-events-design.md`, section 7, owner ruling C1). It runs the
lakehouse's own calls on one input, in two trees, on one machine, and compares the outputs bit for bit.
`c1-record.json` next to this memo holds every number in machine-readable form. The harness is
`harness/c1_record.py`; it is outside the package identity.

## 1. Set-up

| Item | Value |
|---|---|
| Before | `main` at `6eee84ce` (4.144.0); code identity `55f11bdf` |
| After | the frozen branch tree; code identity `c64baa66` (`FREEZE_ID`, `git_tree_hash(paths=CODE_IDENTITY_PATHS)`), copied to the DGX and verified equal; the harness ran from the DGX-local commit `9adae310`, with a clean status |
| Machine | DGX Spark, aarch64, 20 cores (10 Cortex-X925 + 10 Cortex-A725), 119 GB; one venv for both trees |
| Versions | Python 3.12.14, numpy 2.4.4, pandas 2.3.3, scipy 1.17.1, numba 0.65.1 |
| Threads | `OMP_NUM_THREADS=1`; one process at a time; a separate numba cache per tree |
| Input | gradientsports match 10517 from pining, whole match: 254,242 frames, 5,746,254 rows, 1,802 actions |
| Machine load | 1-minute load average during the run: mean 1.3, maximum 2.4 (65 samples, 20 cores) |
| Run | 2026-11-23, the rerun after implementation review round 1 (spec section 7, owner: "Fix all, one full rerun") |

This run replaces the round-0 run on code identity `ba77a3b6` (DGX-local commit `d94022f3`). The round-0 bit compare
was the same in every row of section 2.

The cycle's commit has the code identity `4822c47a`. It differs from this run's `c64baa66` in one line (review
B-81): `silly_kicks/_visible_area.py` `_keys` now copies the array that it writes into
(`to_numpy(dtype=object, copy=True)`). The PR CI found the defect on pandas 3, where a Copy-on-Write `to_numpy()`
array is read-only. No measured run executes that line. The C1 calls, the W5.7 driver and the W8 parser do not reach
the visible-area join, and the W9.6 population passes no `visible_area`, so the join returns before `_keys`. On the
pandas 2.3.3 of these runs, the line gave the same values with or without the copy. The release bump changes only
`silly_kicks/_version.py`, which `CODE_IDENTITY_EXCLUDE` keeps out of the identity. The other files that changed
after the run (tests and documents) are outside the identity.

The calls mirror `luxury-lakehouse` `v2/src/lakehouse/compute/pitch_control.py` and `shape_graphs.py`:

- **Preprocess:** `smooth_frames` + `derive_velocities` with `PreprocessConfig.default()`.
- **Pitch control, loop:** per (period, frame), `compute_pitch_control(grp, attacking_team_id=home)`.
- **Pitch control, batch:** `compute_pitch_control_batch` with one request per frame (same team, `method="spearman"`).
- **Shape graphs, loop:** per (period, frame, team), on the outfield rows (3 or more), `compute_shape_graph(positions)`
  + `infer_positions(graph, positions, 1.0)`. The lakehouse passes the constant direction 1.0.
- **Shape graphs, batch:** `compute_shape_graphs(frames)`. It takes each team-frame's direction from
  `team_attacking_direction`. The harness therefore also runs the loop with that per-frame direction
  (`sg_loop_framedir`), so the batch has an exact reference.

Each pitch-control surface is reduced to its SHA-256, each shape graph to the SHA-256 of its edges, stabilities and
sorted faces. Labels are kept as strings. Nothing is rounded. Timing is wall clock after one warm-up call, so a numba
compile is not counted. Each step ran once.

## 2. Bit compare

| Compare | Equal / total |
|---|---|
| Preprocess columns, `main` against frozen | 17 / 17 |
| Pitch-control loop, `main` against frozen | 254,242 / 254,242 |
| Pitch-control batch against loop, in `main` | 254,242 / 254,242 |
| Pitch-control batch against loop, in the frozen tree | 254,242 / 254,242 |
| Frozen batch against the `main` loop (the lakehouse's switch) | 254,242 / 254,242 |
| Shape graphs, `main` against frozen | 508,484 / 508,484 |
| `compute_shape_graphs` labels against the per-frame-direction loop, frozen | 5,085,212 / 5,085,212 |
| Shape-graph labels with the lakehouse's constant direction, `main` against frozen | 867,965 / 5,085,212 |

The last row is the approved R14 value change (team-relative lateral labels). All 4,217,247 changed labels are the
lateral mirror of the `main` label (L and R, LC and RC swap). No label changes in any other way. The lakehouse passes
direction 1.0 for every team, so after R14 it reads every team as attacking +x.

## 3. Time

One match, one core. "Today" is the lakehouse's calls on `main`. "After, batch" is the frozen tree with the two batch
calls in place of the loops.

| Step | `main` | Frozen | Per frame, `main` → frozen |
|---|---|---|---|
| `smooth_frames` | 0.090 s | 0.089 s | |
| `derive_velocities` | 134.2 s | 41.5 s | |
| Pitch-control loop | 365.9 s | 329.5 s | 1.44 → 1.30 ms |
| Pitch-control batch | 351.0 s | 128.8 s | 1.38 → 0.51 ms |
| Shape-graph loop (constant direction) | 701.2 s | 351.1 s | 2.76 → 1.38 ms |
| Shape-graph loop (per-frame direction) | | 352.1 s | |
| `compute_shape_graphs` | | 299.5 s | 1.18 ms |
| `link_actions_to_frames` (1,802 actions) | 1.15 s | 0.26 s | |

| Lakehouse workload per match | Time | Against today |
|---|---|---|
| Today: preprocess + pitch-control loop + shape-graph loop on `main` | 1,201 s (20.0 min) | 1.0x |
| Frozen tree, the same loops (no caller change) | 722 s (12.0 min) | 1.66x faster |
| Frozen tree, the two batch calls | 470 s (7.8 min) | 2.56x faster |

## 4. Memory

The `tracemalloc` peak of `compute_pitch_control_batch` on the lakehouse workload (one request per frame, 254,242
requests, the default chunk size), frozen tree:

| Backend | Peak | Time under trace |
|---|---|---|
| numba | 7,002,823,212 bytes (7.0 GB) | 198 s |
| numpy | 9,805,360,989 bytes (9.8 GB) | 367 s |

The `compute_pitch_control_batch` docstring states these peaks rounded up (7.1 GB and 9.9 GB). The returned surface
arrays alone are about 3.3 GB (254,242 x 1,600 targets x 8 bytes); the chunk size does not bound them.

Maximum resident set of each whole process (`/usr/bin/time -v`):

| Process | Maximum resident set |
|---|---|
| `main` run (preprocess, both pitch-control calls, shape-graph loop) | 84.3 GB |
| Frozen run (the same, plus both shape-graph batch steps) | 12.9 GB |
| Peak run, numba | 10.0 GB |
| Peak run, numpy | 12.7 GB |

The record does not isolate the `main` maximum to one step. It is consistent with the `main` batch call, which has
no chunk bound and holds the TTI intermediate of every frame at once: 254,242 frames x about 22 players x 1,600
targets x 8 bytes is about 70 GB (inferred, not measured per step).

## 5. Fingerprint cost

`action_fingerprints` (the BLAKE2b binding of each pointer to its action, ruling R9) takes 1.92 ms per match: the
mean of 20 calls on the 1,802 actions, about 1 µs per action.

| Step | Fingerprint cost as a share of the step |
|---|---|
| `smooth_frames` | 2.16 % |
| `link_actions_to_frames` | 0.75 % |
| `derive_velocities` | 0.005 % |
| Pitch-control loop and batch, shape-graph loop and batch | below 0.002 % |

The W0 rule (spec section 4.9) sends a cost above 2 % of any measured step to the owner. The cost is above 2 % only
against `smooth_frames`, which does not compute the fingerprint. The owner accepted the cost on 2026-11-23 (spec
section 7). Each top-level call that takes `links=` computes the fingerprints again, so a pipeline of many such calls
pays the 1.92 ms once per call.

## 6. Against the handoff figures

The spec (section 1) quotes the handoff's profile: shape graphs at about 19 ms per frame, about 81 min for one
gradientsports match. Those frames then held duplicate player rows (lakehouse spec 7.3a). On clean frames, the
lakehouse's shape-graph loop on `main` takes 2.76 ms per frame (11.7 min for this match). `compute_shape_graphs` on the
frozen tree takes 1.18 ms per frame (5.0 min).

## 7. Corpus passes on the frozen tree

Plan Task 15 step 7 and 7b ran again after implementation review round 1, on the same frozen tree as this record
(`FREEZE_ID` `c64baa66`), on 2026-11-23. All passed, with the same results as on the round-0 freeze.

| Pass | Data | Result |
|---|---|---|
| W8 bronze identity | the 7 DFL matches in pining (local run) | the new parser equals the oracle on every match; the string and `datetime64` event times give equal output; no new warning. The new parser logs 0 "Unrecognized GameSection" lines, the oracle 18 (the 18 section-less restart KickOffs) |
| W9.6 game split | SkillCorner 1874553 and 1925299, whole matches, every period, combined in one call (DGX) | 267 of 267 checks equal: each of the 146 public non-exempt members, on the two games combined, equals the same call on each game alone. 121 members run in both link modes (`links` built inside, `links` passed by the caller), 24 take no `links`, and `add_sync_score` requires `links`. Wall clock 4 h 12 min; peak RSS 8.2 GB |
| W5.7 role gate | the 20 public SkillCorner matches (DGX) | passed in all four cells; memo `docs/research/shape-graph-positions/` |

## 8. What this does not measure

- **x86.** This record is arm64 only. Ruling C2 took the HF leg out of the cycle. The x86 proof is CI plus one run of
  the existing `cycle1` parity campaign on the cycle's commit (spec section 7, C2 entry).
- **Spread.** Each step ran once, on one match. The numbers are one sample, not a distribution. The machine was
  shared, at a light load (section 1).
- **Other workloads.** Requests per linked action, many matches in one call, or more than one core are not measured.
- **Lakehouse I/O.** The harness reads one parquet input; it does not time the lakehouse's bronze reads or writes.

## 9. Reproduce

Run from the frozen tree's copy of the harness, with `PYTHONPATH` set to each tree in turn:

```bash
python c1_record.py build-input --provider gradientsports --match 10517 --cache-dir <pining cache> --out <in>
PYTHONPATH=<main tree>   python c1_record.py run  --input <in> --out <out>/main   --label main
PYTHONPATH=<frozen tree> python c1_record.py run  --input <in> --out <out>/frozen --label frozen
PYTHONPATH=<frozen tree> python c1_record.py peak --input <in> --out <out>/frozen --backend numba
PYTHONPATH=<frozen tree> python c1_record.py peak --input <in> --out <out>/frozen --backend numpy
python c1_record.py compare --main <out>/main --frozen <out>/frozen --out <out>/compare.json
```
