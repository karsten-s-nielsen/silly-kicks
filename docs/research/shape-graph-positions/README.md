# Shape-graph positions: the W5.7 real-data role gate

This memo holds the result of the W5.7 role gate of the tracking-batch cycle (spec
`docs/superpowers/specs/2026-11-08-tracking-batch-and-dfl-events-design.md`, section 4.5 W5.7, owner ruling R15). The
gate scores the `infer_positions` labels of `compute_shape_graphs` against the SkillCorner `player_role` of each
player. `metrics.json` next to this memo is the driver's output, byte-identical to the run (SHA-256 `54ad1f4d…`). It
holds only summaries: no row names a player.

**Result: the gate passed.** All three rules pass in all four cells, and `failed_rules` is empty. The fix of the +x
lateral naming (R14) moves the home lateral agreement from 0.041 to 0.935. The other three cells equal the 4.139
baseline, as the W5.6 relations predict.

## 1. Set-up

| Item | Value |
|---|---|
| Driver | `scripts/validate_shape_graph_roles.py` (ADR-052 `for_each`, ADR-128 `PIPELINE`, one unit per match) |
| Command | `python -m scripts.validate_shape_graph_roles run --out <scratch>/roles --cache-dir <pining cache> --workers 8` |
| Code | the frozen branch tree; code identity `c64baa66` (`FREEZE_ID`), the same tree as the C1 record (`docs/research/tracking-compute-perf/`). `run_commit` `9adae310` is the DGX-local commit that holds the shipped copy; `run_tree_dirty` is false |
| Machine | DGX Spark, aarch64, 20 cores, 119 GB |
| Versions | Python 3.12.14, numpy 2.4.4, pandas 2.3.3, scipy 1.17.1, numba 0.65.1 |
| Data | all 20 public SkillCorner matches (`PUBLIC_CORPUS["skillcorner"]`) through pining; periods 1-2, every 10th frame |
| Team-frames | 87,920 per side (labelled, with 3 or more outfield players) |
| Scored players | 583 (292 home, 291 away) |
| Null | 10,000 within-block label shuffles, seed 20261108 |
| Run | 2026-11-23, 1 min 50 s wall clock, 21 of 21 units, 0 failed; the largest process peaked at 2.5 GB RSS |

This run is the rerun after implementation review round 1 (spec section 7, owner: "Fix all, one full rerun"). It
ran after the C5 change of the same day (spec section 7). So the baseline arm is
`scripts/_shape_graph_v4139_baseline.py`, and the driver stamps no oracle digest. The function source of that copy is
byte-identical to the former oracle `tests/oracles/shape_graph_v4139.py`. Both give equal outputs on 400 team-frames in
both directions and on the degenerate cases.

The first run used the round-0 freeze `ba77a3b6` (`run_commit` `d94022f3`), before C5. Its baseline arm was the oracle,
and it stamped `oracle_closure_digest` equal to the frozen constant (`a9102499…`). Its `metrics.json` equals this one
in every field except the code identity, the commit and the two digest fields.

## 2. The four cells

Each agreement is pooled over the 20 matches. The baseline is the 4.139 code, with the direction from the frame label
(the handoff's variant B).

| Cell | Correct / n | Agreement | Wilson 95 % | `null_p95` | `p` | Baseline | Rule 1 | Rule 2 |
|---|---|---|---|---|---|---|---|---|
| Vertical, home | 255 / 292 | 0.873 | 0.830-0.907 | 0.397 | 1.0e-4 | 0.873 (255 / 292) | pass | pass |
| Vertical, away | 255 / 291 | 0.876 | 0.833-0.909 | 0.385 | 1.0e-4 | 0.876 (255 / 291) | pass | pass |
| Lateral, home | 203 / 217 | 0.935 | 0.895-0.961 | 0.553 | 1.0e-4 | 0.041 (9 / 217) | pass | pass |
| Lateral, away | 204 / 217 | 0.940 | 0.900-0.965 | 0.562 | 1.0e-4 | 0.940 (204 / 217) | pass | pass |

- Rule 1: the agreement is above `null_p95`. `p = 1 / 10,001` in every cell, the smallest value that 10,000 draws can
  give: no draw reached the observed agreement.
- Rule 2: the agreement is not below the baseline.
- Rule 3 (mirror invariance, exact): 0 of the labelled team-frames differ from their point-reflected mirror. With no
  failure, there is no face-set flag to report.

The lateral cells score only the players whose role names a side (the first letter of the acronym is `L` or `R`).

## 3. Diagnostics

| Diagnostic | Home | Away |
|---|---|---|
| Tied players (vertical or side mode) | 0 | 0 |
| Team-frames on the pitch-thirds branch | 0 | 0 |
| Team-frames on the lateral equal-frequency fallback | 0 | 0 |
| Of those, with n not in {5, 10} | 0 | 0 |

The pitch-thirds branch and the lateral fallback never fire on real graphs, which confirms the reachability ruling
(ADR-137, owner ruling 2026-11-11).

Confusion counts, final arm: the role acronym against the modal five-level vertical label. "Group" is the truth group
of the committed role table (`B` is back; `DM`, `M` and `AM` are mid; `F` is forward).

| Role | Group | Home B | DM | M | AM | F | Away B | DM | M | AM | F |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CB | back | 4 |  |  |  |  | 7 |  |  |  |  |
| LB | back | 18 | 2 |  |  |  | 16 |  |  |  |  |
| LCB | back | 22 |  |  |  |  | 22 |  |  |  |  |
| LWB | back | 1 |  |  |  |  | 5 |  | 1 | 1 | 1 |
| RB | back | 25 |  |  |  |  | 15 | 3 |  |  |  |
| RCB | back | 22 |  |  |  |  | 23 |  |  |  |  |
| RWB | back | 1 | 1 |  |  |  | 4 | 2 |  | 1 | 1 |
| AM | mid |  |  | 1 | 6 | 7 |  | 1 | 1 | 5 | 6 |
| DM | mid |  | 4 |  |  |  |  | 2 | 1 |  |  |
| LDM | mid |  | 3 | 2 |  | 1 |  |  | 5 |  | 2 |
| LM | mid |  | 4 | 16 | 1 |  |  | 1 | 16 | 2 |  |
| RDM | mid | 1 | 3 | 3 | 1 |  |  | 6 | 3 |  |  |
| RM | mid |  | 7 | 10 |  | 2 |  | 3 | 15 | 2 | 1 |
| CF | fwd |  |  | 2 | 1 | 50 |  |  |  | 2 | 49 |
| LF | fwd |  |  |  |  | 1 |  |  |  |  | 4 |
| LW | fwd |  |  | 1 | 8 | 23 |  |  |  | 8 | 22 |
| RF | fwd |  |  |  |  | 2 |  |  |  |  | 3 |
| RW | fwd |  |  |  | 11 | 25 | 1 |  |  | 6 | 22 |

## 4. The baseline against the handoff's variant B

The handoff (`2026-11-08-silly-kicks-shape-graph-positions-handoff.md`, variant B) measured three matches. The
baseline arm of this run gives the same count in every cell:

| Match | Home vertical | Home lateral | Away vertical | Away lateral |
|---|---|---|---|---|
| 1886347 | 13 / 13 (handoff 13 / 13) | 0 / 7 (0 / 7) | 10 / 14 (10 / 14) | 8 / 9 (8 / 9) |
| 1899585 | 13 / 14 (13 / 14) | 2 / 11 (2 / 11) | 11 / 14 (11 / 14) | 9 / 11 (9 / 11) |
| 1925299 | 12 / 15 (12 / 15) | 1 / 12 (1 / 12) | 10 / 15 (10 / 15) | 8 / 11 (8 / 11) |

The frame path is different: the handoff used the lakehouse v2 path (`shape_skillcorner_input` and
`convert_to_frames`), and this driver uses `_loader_pining.build_skillcorner_frames`. The population predicate is the
handoff's, and every denominator is equal. So there is no difference to explain.

On the same three matches, the final arm gives 7 / 7, 9 / 11 and 11 / 12 for the home lateral cell. The other three
cells equal the baseline.

## 5. The away vertical gap

The handoff found the away vertical agreement (31 / 43 over its three matches) below the home agreement (38 / 42), and
left the cause open. On all 20 matches the gap is gone: away 255 / 291 (0.876), home 255 / 292 (0.873).

Per match, the away agreement minus the home agreement has a mean of 0.000 and a standard deviation of 0.129 (minimum
-0.286, maximum +0.200). The away agreement is lower in 7 matches, higher in 7 and equal in 6. So the gap in the three
handoff matches is inside the match-to-match spread. This work found no defect, and no code changed for it.

The vertical misses have the same classes on both sides:

- A winger (`LW`, `RW`, truth forward) gets `AM` or `M`, which maps to mid: 20 home, 14 away.
- An `AM` (truth mid) gets `F`: 7 home, 6 away.
- A `CF` gets `AM` or `M`: 3 home, 2 away.

The away teams in this set field more wing-backs (16 against 3), and 9 of the 16 away wing-backs score correct. This
is a role-mix effect on the per-match numbers (inferred, not proven).

## 6. What the agreement does not measure

The reference is the role that SkillCorner assigns to a player for the match: a nominal formation role. The shape
graph labels the shape that the team takes in each frame. A winger who holds the midfield line in the team's shape gets
`AM`, and the role table counts that as a miss. So an agreement below 1.0 is not only label error. The permutation
null controls for each team's role mix, but not for the gap between the nominal role and the shape a team plays.

The gate tests the label convention and the direction handling on real data. It does not validate the labels as a
measure of a tactical role.
