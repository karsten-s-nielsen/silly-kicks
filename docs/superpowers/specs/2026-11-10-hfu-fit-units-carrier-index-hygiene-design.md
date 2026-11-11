# HFU — xsuccess fit units, ball-carrier pre-index, campaign hygiene — design

Branch `feat/hfu-reuse-fit-carrier` (off `main` at `bb0c601c`, silly-kicks 4.140.0). One code commit for the cycle;
one more commit only for launch evidence (provenance). Every commit, push, force-push, PR, merge, tag and HF job is
owner-gated. Scope owner: Karsten. Reviewers A and B (independent); reports in
`D:\Development\_reviews\2026-11-10-hfu-fit-units-*`. Status: revision 3b (3 plus two changes from the plan review,
both owner-confirmed 2026-11-10: §3.2 "Removed" and the `lost_output` field of C4). Round 1: A and B REQUEST CHANGES. Round 2
(revision 2): A and B APPROVE, no blocking finding. Revision 3 applies their round-2 notes and two owner rulings
(§12, §13). Revision 3c (2026-11-10): the build notes (§14), four owner rulings from the implementation review
(§4.2, §4.5, §15) and the answers to implementation review round 1 (§15). Revision 3d (2026-11-11): §17, the
record of launch 8 and two owner rulings that it caused (the `compare` baseline; the Hub request count), and a
third ruling on a test order dependence that the final check found (§17.5). The second commit holds them
together with the evidence; the cycle commit `abcfda86` is not amended. Revision 3e (2026-11-11): §18, the
answers to implementation review round 5 and three more owner rulings.

## 1. Why

The 4.140.0 cycle left a list of optimizations and open issues (`ADR-128` amendments 2026-11-02 to 2026-11-10;
the cycle ledger; the handoff `D:\Temp\hfu-change11\handoff-next-hfu-cycle.md`). The owner set the scope on
2026-11-10 after a read-only measurement of launch 7. This spec states the standing decisions only.

### 1.1 The measure that decides

`cpu-upgrade` costs $0.03 per hour (`docs/howto/hf-jobs-training.md`). Launch 7 used 1,810 job-minutes: about
$0.90. Job-minutes are almost free. An HF change is ranked by wall time per launch, by failed launches and by
storage. It is never ranked by job-minutes.

### 1.2 Evidence (launch 7 on `a1a3b52f`, 388 jobs: 379 COMPLETED, 7 ERROR, 2 CANCELED)

Source: HF job records and logs, read-only. Files: `D:\Temp\hfu-next\measure\` (`jobs_a1a3b52.json`,
`logs_a1a3b52.json`, `analysis.txt`, `peek.py`, `peek2.py`, `peek3.py`, `hpo_job.log`, `bucket_size.log`,
`org_storage.txt`, `org_storage2.txt`, `cmp_launch6_7.txt`, `cmp_rule_v2.py`, `cmp_rule_v2_6v7.txt`,
`cmp_rule_v2_7v7.txt`), `D:\Temp\hfu-next\closures\` (`measure.txt`, `measure2.txt`) and
`D:\Temp\hfu-next\prune\delete.log`.

| Id | Observation |
|---|---|
| M1 | Wall time 4 h 35 min, with a storage stop (the handoff gives 35 min; no log was read for it). The long pole is xsuccess: `refs` + `extract` 25 min, the `hpo` job 143 min (one unit, 8,470 s for 30 trials: trials 1-10 1,535 s, trials 11-30 6,935 s; the stage took 187 min of wall with the stop), the `fit` job 63 min (one unit, 3,685 s). Every other instance ends inside that window. |
| M2 | Scheduling wait: median 4 s. Job setup (clone + `uv sync`): median 19 s, 152 job-min in total. `Prepared 57 packages`: median 4.4 s. |
| M3 | A job spends about 64 s outside its unit (110 jobs with one unit or none; `peek.py`). Cause read from code, not probed: `_write_confirm.JOB_SETTLE_S = 30`, once in the worker and once in the reduce subprocess. |
| M4 | 15 stage generations were reduced by several jobs: 7,284 of 8,374 reduce seconds (`peek3.py`). All ended; each of the four public `extract` reduces took about 145 s. |
| M5 | On 2026-11-10 the scratch bucket held 23.3 GB; one campaign is 7.3 GB. The 12 launch-6 run keys (`cycle1-0552f58-*`, 7.28 GB) were then deleted on the owner's yes; the bucket holds 16.1 GB. The org's six private datasets hold 28.4 GB (`usedStorage`); one of its ten private models was measured (2.3 MB). Second measurement, later on 2026-11-10: the private datasets hold 22.07 GB, after the TF-46 session deleted 6.36 GB of old dataset generations on the owner's word (the revisions that `datasets.lock.json` pins are untouched). |
| M6 | The static import closure of every pipeline module holds the same 178 files: 157 core library files (`spadl`, `tracking`, `vaep`, `win_probability` and `xthreat` import each other) and 21 runner files. Each pipeline adds 2 to 13 files. |
| M7 | Launch 6 against launch 7 under the §6 rule (`cmp_rule_v2.py`): 16 of 21 artifacts identical, the xsuccess artifact and all five ghost_gk artifacts among them. The 5 differences: `metrics.json` of the four xShot/xCross `public` artifacts (the launch-6 fix F30 changed what they record) and `census.json` (the census is compared by `verify` with its tolerance). Launch 7 against itself: 21 of 21. Both reviewers ran the rule as §6 words it and got the same 16 of 21. One leaf, `artifact_size_bytes`, changed by 1 byte between the launches with no code change (xCross `sc_extended` `position_only`: 1,663,724 against 1,663,725): it counts the bytes of `metadata.json` (2,012 against 2,013), and the kernel version in `training_platform` changed length. |
| M8 | `_pre_index_frames` (the ball-carrier pre-index) takes 13.5 to 58.0 s per match on HF (audit 2026-10-30, job `6ae4bd0f0e0338d1bb7cbdf2`): 58 s of a 115 s ghost_gk extract unit at 30 Hz. It builds one DataFrame per frame (`silly_kicks/tracking/_ball_carrier.py:109`). The ghost_gk `extract` unit mean at launch 7 was 91.6 s over all variants (88 s for `default`). |

### 1.3 Scope (owner 2026-11-10)

Each row is an owner ruling of 2026-11-10 in the session that wrote this spec; the record is memory
`project_hfu_next_cycle_scope.md`.

| Handoff item | Decision | Reason |
|---|---|---|
| 4 xsuccess `fit` as stage units | IN, Part A | `fit` is 63 min of the long pole (M1). |
| 10a ball-carrier pre-index | IN, Part B | It serves every `infer_ball_carrier` caller in the library (M8). A wheel change is accepted. |
| 8 read bound in `retrieve` | IN, Part C1 | HTTP 429s during retrieve. |
| 11 `post_verify` without a Hub class | IN, Part C2 | `xsuccess-v1` was verified by hand. |
| 12 storage: preflight check and prune | IN, Part C3 | Launch 7 stopped on the org storage limit. |
| 13 FileSet member check | IN, Part C4 | Over the limit, a result kept its marker and lost its members. |
| 14 STUCK operator path | IN, Part C5 | A repaired unit stayed STUCK until the driver restarted. |
| 15 card citation gate, `.gitattributes` | IN, Part C6 | The open `TODO.md` row. |
| The spec §14.10a record | IN, Part C7 | The owner approved it with the author's calls ("the spec §14.10a paste goes into this cycle's branch"). |
| 1 reuse of stage results across commits | DROPPED | M6: a sound derived closure reruns everything on any runner or core-library edit, where campaign fixes land. A per-function variant is too complex. PR #290 and the ADR-129 stub stay untouched. |
| 2 job setup: prebuilt image, `uv` wheel cache, `xgboost-cpu` | DROPPED (the whole item) | M2: at most 11 s per job. |
| 3 stage chaining; the settle time (M3) | DROPPED | About 8 and 7 min, off the long pole (M1). The settle guards against lost writes. |
| 5 hpo startup batch; 6a hpo startup over jobs | DROPPED | M1: about 8 min of 143; the TPE phase has no byte-identical speed-up. |
| 6b `fit` units over jobs | Delivered by item 4 | The driver shards stage units. |
| 7 reduce lease | DROPPED | M4: no wall time and no measured harm. The bucket mount gives no reliable lock. |
| 9 extraction and launch dedupe (all four parts: the superset extraction, `position_only` as a column subset, the public-tier units equal to the `sc_extended` units, the four identical xCross `probe` runs) | DROPPED | The re-measure is M1: no part is on the long pole. Owner: "confirmed, if no walltime savings". |
| 10b per-frame feature loops | DROPPED | No wall time (M1); a float bit risk. |
| D1-IMPL-09, -11, -12 | Accepted limits | ADR-128 "Known limits"; no measured cost at launch 7. |

No `TODO.md` row is opened for a dropped item (owner).

### 1.4 Expected result

The `fit` stage goes from 63 min to about 12 min (§3.4), so a complete run goes from about 3 h 50 min to about
3 h. `hpo` stays at 143 min. Parts B and C do not change campaign wall time.

## 2. Rules for every part

- **No output changes.** Parts A and B must not change one byte of any artifact payload. §6 proves it at
  production scale.
- **No new mechanism where a seam exists.** Part A uses the stage contract and the driver's own sharding. Part C2
  uses `load_check`'s loader map. §6 extends `scripts/hf_jobs/identity.py`.
- **Tests first.** Each test in §7 is seen to fail before its change.
- **One code commit.** No micro-commits. No commit without the owner's yes.

## 3. Part A — xsuccess `fit` as stage units

### 3.1 Today

`XSuccessModel.fit` with the `xgboost` family runs 6 xgboost fits (`silly_kicks/xsuccess/_model.py:190`): the model,
and 5 `cross_val_predict` fits (`StratifiedKFold(n_splits, shuffle=True, random_state=seed)`) that feed one
isotonic fit. The trainer's `fit` stage runs 6 such calibrated fits: 5 GroupKFold-by-match folds and the final fit
(`scripts/train_xsuccess.py` `_fit_map`). So the one `fit` unit holds 36 xgboost fits. They run on the InnerPool
3 wide, because one calibrated fit peaks at 7.26 GiB (HFU spec §16.7).

### 3.2 Change

The 36 xgboost fits become 36 units of the `fit` stage. The driver shards them over jobs. No stage is added, and
`refs`, `extract` and `hpo` do not change in code or in output.

**The unit grid is fixed:** `<fit>-main` and `<fit>-cal0` to `<fit>-cal4`, for `<fit>` in `final`, `fold0` to
`fold4`. `fit.units(ctx)` returns these 36 keys and reads nothing, so a `--plan` reads no large file.

**The row space is the pooled rows.** Every unit and the reduce read the `extract` output (the pooled rows, 75 MB)
and call `_cv_splits(pooled)`, as `_fit_work` does today. The GroupKFold folds are computed on ALL pooled rows.
The finite-feature filter runs after the fold selection, inside the fit steps, as `XSuccessModel.fit` does today
(`_model.py:167-172`). Launch 7 has 7,974,436 pooled rows and 48 of them have a non-finite feature.

**What a unit does** (`<fit>`'s rows = all pooled rows for `final`, fold `i`'s train rows for `fold<i>`):

| Unit | Work | Result |
|---|---|---|
| `<fit>-cal<j>` | From the fit's kept rows, take calibration split `j` (the library's split step); fit on its train rows; `predict_proba` on its test rows. | A one-column DataFrame, `p` float32, in test-row order. |
| `fold<i>-main` | Fit the booster on the fit's kept rows; predict RAW (before isotonic) on fold `i`'s test rows; a test row with a non-finite feature gives NaN. | A one-column DataFrame, `raw` float64: the vector `predict_success` holds before its isotonic step (`_model.py:263-266`). |
| `final-main` | Fit the booster on the kept rows of all pooled rows; save it; predict RAW on the chirality probe rows with the in-memory booster. | A `FileSet`: `model.json` (the bytes `save` writes) and `probe_raw.npy`. |

- A parquet column keeps its dtype and its bytes, so no result crosses a JSON float round trip.
- **A unit that does not exist** for the input returns the JSON value `{"absent": true}`: a `cal<j>` beyond the
  fit's calibration split count (no calibration applies, or fewer than 5 splits), a `fold<i>` beyond the fold
  count, and every `cal` unit of the `per_type_logistic` family. On the campaign corpus all 36 units exist.
- **`per_type_logistic`:** `<fit>-main` fits the whole model (`_fit_per_type` has no calibration). `final-main`
  returns `model.json` as `save` writes it and the probe prediction as `probe_raw.npy` (no isotonic step follows).
  `fold<i>-main` returns `predict_success` on the test rows.
- **Fewer than 2 matches:** there are no folds. `final-main` also returns `in_sample_raw.npy`, the raw prediction
  on every pooled row, and the reduce scores it in sample, as `cross_val_metrics` does today
  (`train_xsuccess.py:44-45`).

**The reduce** reads the 36 results in sorted key order and the `extract`, `refs` and `hpo` outputs. For each
calibrated fit it places the `cal` predictions in the fit's out-of-fold vector by the same calibration splits,
and takes the isotonic thresholds. It calibrates each fold's raw predictions, assembles the out-of-fold vector
over all pooled rows in fold order, and computes the metrics with `_oof_metrics(pooled, y, oof)`. `n_rows`,
`n_matches` and the per-type table come from the pooled rows, as today. It builds `metadata.json` (the final fit's
thresholds; the probe prediction = the calibrated `probe_raw`), `metrics.json`, `MODEL_CARD.md` and `SHA256SUMS`
as today. It never reloads a booster and never calls xgboost.

**Removed:** `inner_parallel`, `inner_width` and `timeout_s` on the `fit` stage; `_fit_task`, `_fit_map`,
`_FIT_ROWS`, `_fit_work`; `cross_val_metrics`, which only the serial path used (it moves to the tests as part of
the oracle), and with it the `[progress] xsuccess fold <k>/<n>` line of HFU spec §15.3; `_fit_map` takes the
`[progress] xsuccess fit <k>/<n>` line of HFU spec §16.2 with it. Each unit prints its `[unit] fit/<key> done in
<s> s` line instead (owner 2026-11-10: "Option 1 confirmed"); the `fit` part of `tests/scripts/test_xsuccess_drivers_smoke.py:158-222` is rewritten for the units;
the `cycle1.toml:32-33` comment follows.

**Library seam.** `XSuccessModel._fit_xgboost` and `predict_success` are split into the steps the units and the
reduce need: fit one booster; give the calibration splits; fit one calibration split and predict its test rows;
take the isotonic thresholds from an out-of-fold vector; predict raw; calibrate raw; build the metadata payload
from its parts. `_fit_xgboost`, `predict_success` and `to_dict` compose the same steps, so a library caller and
the trainer run ONE composition. The public signatures of `fit`, `predict_success`, `to_dict`, `save`, `load` and
`bundled` do not change. The step names are the builder's choice.

### 3.3 What must hold (the byte-identity risks)

| Id | Risk | Gate |
|---|---|---|
| A-R1 | The explicit calibration loop differs from `cross_val_predict` (order, dtype of the out-of-fold vector). | Unit test: the new `_fit_xgboost` equals the old one (kept as a test oracle): `model.json`, the isotonic thresholds and `predict_success`, byte for byte. The test asserts that calibration ran with 5 splits (`_iso_x` is not None). It also runs with 2 to 4 splits, with fewer than 50 rows and with one class. If it cannot be made equal, stop for the owner. |
| A-R2 | The folds are computed in another row space than today. | §3.2 fixes the space. The reduce test's toy corpus holds at least one row with a non-finite feature and matches of unequal row counts; the test asserts both are present (non-vacuity). |
| A-R3 | A result loses its dtype in the store, or a small shape takes a wrong branch. | The reference is today's `_fit_work` on its serial path (`cross_val_metrics`, `fit`, `save` and the assembly of `metadata.json`, `metrics.json` and `MODEL_CARD.md`, `train_xsuccess.py:296-340`), kept in the tests as the oracle. The reduce test compares the new stage output with it THROUGH the store (`--run-all`), every file byte for byte after the provenance keys are removed. It runs once for each shape: the calibrated xgboost corpus of A-R2, fewer than 5 calibration splits, no calibration (one class, or fewer than 50 rows), `per_type_logistic`, and fewer than 2 matches. Launch 8 runs none of the small shapes, so this test is their only gate. |
| A-R4 | The chirality probe is predicted from a reloaded booster. | `final-main` predicts the probe with the in-memory booster (§3.2). The reduce test compares `metadata.json` byte for byte. |
| A-R5 | A unit's thread count differs from an InnerPool child's. | Both are pinned (`scripts/_thread_pin.py`). The parity test runs W1N1 against W2N2. |
| A-R6 | The bundle differs from launch 7. | §6: `model.json`, `metadata.json` and `metrics.json` equal launch 7's under the §6 rule (M7: launches 6 and 7 were equal). `metrics.json` is the production witness of the 30 fold fits. |

### 3.4 Memory and sizing

- One unit does less than one calibrated fit on the same rows: it holds the pooled rows, the features of its own
  rows and one booster. The measured peak of a whole calibrated fit, 7.26 GiB (HFU spec §16.7), is therefore used
  as the upper bound of one unit. This is a bound by reasoning, not a measurement.
- `cycle1.toml` sets `"xsuccess/fit" = 8589934592` under `peak_rss_by_stage` (8 GiB, the round value ghost_gk
  uses: the 7.26 GiB bound plus a margin, because the bound is one sample). A job then runs 3 units at once, as
  today (`size_workers`, `executor.py:395`). A unit over the cap is deferred and reruns alone (`--escalate`), as
  for every stage; its bytes are the same and the stage takes about 25 min more.
- **New campaign knob `units_per_job_by_stage`** (a `"job/stage"` map, like `peak_rss_by_stage`): `"xsuccess/fit"
  = 3`. The stage then splits into 12 jobs of 3 units (`max_shards` is 12), and all 36 fits run at once. Without
  the knob the global `units_per_job = 20` gives 2 jobs and 6 fits at once (about 32 min). The knob applies to the
  keyed launches and to the round-robin launches (`_keyed_launches`, `_round_robin_launches`); an escalate job
  stays one unit worker.
- Expected wall time: the longest fit (about 430 s, from the 2,561 s of the final calibrated fit) plus the job
  overhead and the reduce: about 12 min. The 36 fits total at most 11,055 core-seconds (3 × 3,685 s at launch 7).
- The unit ledgers of launch 8 record the real peak RSS.
- The reduce is heavy for a reduce (six isotonic fits, the metrics, the card) and runs in each job that finds
  the stage complete (M4; the lease is dropped). Each such reduce holds the pooled rows (about 1.2 GiB) and the
  feature matrix of all pooled rows for the finite test (about 2.2 GiB, `_model.py:171`).

No probe job is needed, so the cycle has one code commit.

### 3.5 Registry, toy input, parity test

`scripts/_toy_corpora.py` (the non-finite row and the unequal matches of A-R2), `tests/scripts/test_parity_xsuccess.py`
and `tests/scripts/test_xsuccess_drivers_smoke.py` follow the new unit grid. The parity test asserts `outputs`
and `artifact` equal between `--run-all --workers 1` and 2 jobs × 2 workers, with `completion_out_of_order` for
the `fit` stage. The instance args do not change, so `registry.py` is expected to need no change (the plan
checks it).

## 4. Part B — the ball-carrier pre-index

### 4.1 Today

`_pre_index_frames(frames) -> dict` (`silly_kicks/tracking/_ball_carrier.py:38`) turns long-form frames into dense
arrays. It builds `dict(iter(player_rows.groupby(...)))` (one DataFrame per frame, line 109) and copies each
group in a Python loop, with an inner loop per player for the slot ids. `infer_ball_carrier` and
`silly_kicks/calibration/_carrier_objective.py:65` are its two callers. `infer_ball_carrier` is called across the
library: `tracking` (`features.py`, `_ghost_gk.py`, `_ghost_outfield.py`, `_xshot_occurrence.py`,
`_xcross_attempt.py`, `_model_eval.py`), `gkdv`, `positioning`, `restdefense`, `coordination`, `causal` and
`calibration`. `atomic/tracking/features.py` re-exports `ball_carrier_at_action`.

### 4.2 Change

Rewrite ONLY the placement of the player rows into the dense arrays (today lines 108-138): `px`, `py`, `pvx`,
`pvy`, `player_slots`, `n_valid` and `max_players`, without a per-frame DataFrame and without a per-player Python
loop. Everything else stays as it is: the ball/player split, the slot maps (lines 62-74), the ball aggregation,
the unique-frame table, the segment bounds and the returned metadata. The function name, its signature and every
key, dtype and value of the returned dict do not change.

The rewrite does no float arithmetic: it only places values.

Two more changes inside the function, both with no output change (owner rulings 2026-11-10, implementation
review): the frame-lookup dict (former lines 94-97) is removed, because the replaced loop was its only reader;
and the two remaining `groupby` calls (the ball aggregation and the segments) state `observed=True` (§4.5).

### 4.3 What must hold

The current function moves, unchanged, into the tests as the oracle. The new function must equal it on every key.

**Equality rule.** Arrays: same dtype and shape, and equal element by element where two NaN values are equal; an
object array is compared element by element with the same NaN rule. Lists and dict keys and values: the same
NaN-aware comparison, in order. Dtypes: `==`. **An input on which the oracle raises must make the new function
raise the same exception type.** As built, a float array is compared BY BITS, which is stricter: `-0.0` differs
from `0.0` (implementation review, A-09).

**Semantics that the oracle fixes** (the rewrite must keep each; rows marked "unchanged code" are outside the
rewrite and listed so the plan does not touch them):

| Id | Semantics |
|---|---|
| B-S1 | A player row is a row that `~(is_ball == True) & x.notna()` selects. With an object `is_ball`, a null value makes a player row; with a nullable `boolean` `is_ball`, a null value drops the row (reviewer A's probe, pandas 2.3.3). (unchanged code) |
| B-S2 | Inside a frame the slot order is the input row order. |
| B-S3 | A player row whose `game_id`, `period_id` or `frame_id` is null is not placed (pandas `groupby` drops it); its frame has `n_valid = 0`. The ball aggregation keeps null keys (`dropna=False`). |
| B-S4 | `max_players` is the largest frame's player count, 1 when no player row is placed. |
| B-S5 | `player_slots` holds the slot of the row's `player_id` by `dict.get`, and -1 where a slot is empty OR where the id is not found. A float NaN `player_id` is never found: the row takes a slot, counts in `n_valid` and has `player_slots == -1`. |
| B-S6 | A duplicate `player_id` in one frame gives two slots with the same slot id. |
| B-S7 | Without `vx`/`vy`, `pvx`/`pvy` are 0.0 in the valid slots and NaN elsewhere. |
| B-S8 | Coordinates are float64 in the arrays whatever the frame storage dtype (ADR-106: float32 storage, `category` `team_id`). |
| B-S9 | The slot universe is `sorted(player_rows["player_id"].unique())` over ALL player rows, a row with a null frame key included; `slot_to_team_id` is the team of the id's first row in input order. A nullable `Int64` id with `<NA>`, or an object `None` beside strings, on a player row raises `TypeError` there. (unchanged code) |

**Oracle inputs:** every fixture of `tests/tracking/test_ball_carrier.py`, `test_ball_carrier_numba_parity.py` and
`tests/scripts/test_coordination_driver_modules.py`; each provider's toy corpus with its real id dtypes
(`_toy_corpora`, `PROVIDER_ID_KIND`); a synthetic 20,000-frame match; and adversarial frames: a null frame key; a
`player_id` that occurs only on a null-frame-key row; a float NaN `player_id` on a player row; an `Int64` `<NA>`
and an object `None` `player_id` on a player row (the oracle raises); a duplicate player id in one frame; two ball
rows in one frame; a frame with no player; a null `is_ball` in an object column and in a nullable `boolean` column; frames without
velocity; `category` `team_id` and
`category` frame keys; float32 coordinates; unsorted input rows; an empty frame table.

`infer_ball_carrier`'s output must also be equal on the same inputs.

### 4.4 The production check (a one-time read-only HF job, owner go)

The handoff bar is a parity proof at production scale. Launch 8 gives it indirectly, for four trainers only. So
ONE measurement job runs the oracle and the new function on one complete real match of each tracking provider in
the staged `tc3` dataset (Gradient Sports at 30 Hz, Sportec at 25 Hz, SkillCorner at 10 Hz: three matches, the
real id dtypes) and asserts the §4.3 equality for each. It prints the seconds of each function per match, which
is the production speed measurement.

- **It runs once, for this rewrite.** It is not a campaign step and it never becomes part of normal operations
  (owner 2026-11-10). The standing guard is the oracle unit test in CI.
- Mechanism: the measurement-job pattern of the 2026-10-30 audit (`D:\Temp\hfu\audit\launch_audit_probe.py`): one
  `cpu-upgrade` job at the pushed cycle commit, the campaign image and install, the dataset mounted read-only, no
  store write. The oracle comes from the cloned `tests/`.
- The job script is kept outside the repo. It goes to both reviewers with the implementation; its sha256 and its
  full output go into the evidence commit (§6 step 5).
- Non-vacuity, asserted by the script: exactly one match of each of the three providers was loaded and compared;
  the oracle and the new function are distinct objects; a match that cannot be loaded fails the job.
- Cost: about 5 min. Order: after the cycle commit is pushed and CI is green, before launch 8.
- A failure stops for the owner. The fix is an amend and a force-push, each on its own owner yes.

### 4.5 Speed and library effects

- Speed is not asserted in a test (a timing assertion is flaky). The plan records the before and after seconds
  on the synthetic 20,000-frame match on this PC; §4.4 gives the production figure for three matches.
- If any per-frame Python loop remains, it carries an ADR-073 scale guard. As built: no such loop remains, and
  the one pass over the placed rows has a growth guard in the registry (`SCALE_GUARDED`,
  `tests/test_scale_guards.py::test__pre_index_frames_is_subquadratic`).
- On `category` frame keys the function emitted a pandas `FutureWarning` (`observed=False`) from three `groupby`
  calls: the player groups (line 109), the ball aggregation and the segments. The rewrite removes the player
  `groupby`. The two others state `observed=True` (owner ruling 2026-11-10): the pandas 3 default and the
  ADR-112 convention, with equal output on every oracle input (`category` keys with unused categories and with
  nulls included). So the function gives no such warning; a test asserts it. This is the one observable change;
  `CHANGELOG.md` states it.
- `docs/PRIVATE_CONSUMERS.md` has no row for `_ball_carrier` (checked). The `pre=` argument of `infer_ball_carrier`
  keeps its meaning. `CHANGELOG.md` records the change as a speed-up with no output change.

## 5. Part C — campaign hygiene

### C1 — one read bound in `retrieve`

Today `retrieve` runs 4 instances at once and each `_counters` call opens its own 8-thread pool
(`scripts/hf_jobs/adopt.py:150`, `:197`): up to 32 concurrent store reads. Change: `retrieve` owns one bound of 8
concurrent store reads, and EVERY store read of `retrieve` goes through it (`_counters`, the output read, the
`_inputs.json` reads, the eval rows). The merge stays in sorted key order. Test: a counting fake store never sees
more than 8 reads in flight with 4 instance workers; the totals are unchanged.

### C2 — `post_verify` for an artifact without a Hub class

Today `post_verify` calls `<class>.from_hub(repo)` for every Hub artifact, with the class from `LOADERS[a.job]`
(`adopt.py:1198`). Only four classes have `from_hub` (`GhostGkModel`, `GhostOutfieldModel`, `XShotOccurrenceModel`,
`XCrossAttemptModel`). The manifest entry `hub_xsuccess` declares `hub_class = ""`
(`docs/artifacts/manifest.toml:551`).

Change: an artifact with a non-empty `hub_class` round-trips with `from_hub`, as today. An artifact with an empty
`hub_class` round-trips with the job loader: the published files are downloaded to
`<staging>/_post_verify/<repo name>/` and loaded with `load_check` (the `LOADERS` map). Tests: a manifest gate
(every Hub artifact with a non-empty `hub_class` has a `LOADERS` class of that name with `from_hub`; every one
with an empty `hub_class` has a `LOADERS` class with `load`), and both branches of `post_verify` with a fake API.

### C3 — storage: a preflight check and a prune command

- **Preflight check 7, `storage`.** Used = the sizes of the org's buckets (`bucket_info().size`) plus `usedStorage`
  of each private dataset and private model of the org. The check fails when `used + storage_headroom_bytes >
  storage_limit_bytes`. Both values are in the campaign file: `storage_limit_bytes = 100_000_000_000` (100 GB,
  owner-verified 2026-11-10: the owner read the private limit off the HF billing page; the public limit is
  7 TB), `storage_headroom_bytes = 20_000_000_000`. One
  campaign is about 7.5 GB after Part A (the 36 unit results add about 0.2 GB), so the headroom is two campaigns
  and 5 GB. The check prints the used bytes by repo. Measured today: 16.2 GB of buckets and 28.4 GB of datasets
  (22.07 GB at the second measurement of M5); nine private models are not measured (the one measured is
  2.3 MB). It is not verified that these two API
  figures are what HF counts against the limit; the check is a guard, not the limit itself.
- **`campaign <file> prune`.** Without a flag it lists, for this campaign, the run keys of every commit other than
  the local HEAD, with their file count and bytes, and deletes nothing. `prune --delete
  --i-have-owner-approval <commit7>` (the flag once per commit, as `publish` names each repo) deletes the run keys
  of the named commits (`batch_bucket_files(delete=...)`).
  - It refuses the local HEAD's commit, a commit with a non-terminal job, and any key that is not exactly
    `<campaign>-<commit7>-<job>-<tier>` (or with `-r<N>`) for a job of this campaign. A campaign whose name is a
    prefix of another campaign's name does not match that campaign's keys.
  - It never touches `_campaign`, `_staged`, `_canary` or another campaign's keys.
  - For a commit that `docs/artifacts/manifest.toml` names as a `training_commit`, it prints a notice with the
    artifact count before it deletes (the scratch keys are run records; the artifacts are in the wheel and on the
    Hub).
  - After each run key it lists the key again and prints the files left. It exits non-zero when a file is left.
    On a `403` it stops, prints that the org is at its storage limit (at the limit HF refuses the bucket delete
    too) and names what the operator can delete from another source. A second run is safe: it deletes what is
    left.
  - Stop the driver before a prune: between two job waves no job is non-terminal. The howto says so.
- Deletion is irreversible and needs the owner's yes for each run. The howto says when: after an amend is pushed,
  and after a launch is retrieved and verified.
- Tests: the check on a fake API (under, at and over the bound); `prune` lists and deletes exactly the named
  commits' keys; each refusal; the prefix-name case; a partial delete and a `403`.

### C4 — a `FileSet` result is present only with its members

Over the storage limit the mount accepted writes locally, the small files landed and the large uploads were lost.
A `FileSet` result then had its `<key>.fileset.json` marker and no `<key>.files/` members; the controller's plan
read the unit as done and the stage as reduced, and a downstream unit failed with `prep/only is not a FileSet
result`. A lost `.parquet` or `.json` result heals itself: its name is absent from the listing, so the unit is
pending.

Change: `plan()` counts a `FileSet` result (a unit result or `_output`) as present only when every member its
marker names is in the store listing. A unit whose members are missing is `pending` and is named in a new plan
field, `lost_units`; a stage with such a unit, or with such an `_output`, is not `reduced`, and `ready` uses the
same rule.

- **In a running driver:** the unit is a candidate again and reruns while its coverage is below `attempts`
  (`driver.py:655-658`; the jobs that wrote the lost result count toward it, `:1005-1010`). When its coverage is
  spent, the report says `<stage>/<key>: result lost (members missing); free storage, stop the driver, then run
  resume`, never "crashed with no record". After a `resume` (a new baseline) the unit reruns in every case.
- **A lost `_output`.** The stage record also gets `lost_output` (true when `_output` is a `FileSet` whose
  members are not all present). The driver then triggers the reduce again, as for any stage that is not
  reduced. When the reduce triggers are spent, the report says `<stage>: output lost (members missing); free
  storage, stop the driver, then run resume`.
- When the unit result is rewritten and the `_output` is still valid, no reduce runs (`executor.py:310`); the
  rerun gives the same bytes, so the output stays right.
- **Out of scope:** a member that is present and corrupt. A downstream unit fails on it, as today.
- The check runs in `plan()` only, where the store is the bucket API's one recursive listing. In-job code is
  unchanged (the writer's own mount always shows its files).
- Cost: one small marker read per `FileSet` result per plan (memoized per `--plan` process). Part A adds one
  `FileSet` unit to xsuccess (`final-main`); the other 35 results are parquet. The plan measures the request
  counts on the four instances of HFU spec §16.4 and records them; a count above today's count plus the number of
  `FileSet` results of the instance is a defect. **Corrected 2026-11-11 (§17.4):** this bound counted a marker
  read as one HTTP request; it is two. The bound is in store operations: one marker read per `FileSet` result.
- The plan checks `scripts/hf_jobs/datasets.py:359-361` (the download plan with `verify=False`) for a
  `uses_dataset` stage that this rule changes.
- Tests: a store with a marker and a missing member gives a pending unit, `lost_units` and an un-reduced stage;
  some members missing; a member-less `_output` beside complete units; a complete result is unchanged; a
  `.parquet` result is never read by the check; and a `decide`-level test on the launch-7 state (a marker without
  members, a completed job in the coverage cache, a stale `_output`) with coverage below and at `attempts`.

### C5 — the operator path for a STUCK unit

The driver counts the failed jobs it saw since its start (the baseline rule, ADR-128 amendment 2026-11-02), so a
unit that the operator repairs stays STUCK in that run. This is the design. Change: the STUCK report line ends with
`fix the cause, stop the driver, then run resume`, and `docs/howto/hf-jobs-training.md` gains a short section that
says the same and why. No behaviour change. Test: the report line text.

### C6 — the card citation gate and `.gitattributes`

The open `TODO.md` row. Its "today" sentence is stale: `tests/test_model_cards_match_bundles.py` takes its
populations from the artifact manifest (lines 22-32), and no test checks that a file a card cites exists. The
defect it guards against: a bundled card that cites `metrics.json` while the bundle has none.

- (1) A new test. Population: every `MODEL_CARD.md` under `silly_kicks/` (derived from the tree, ADR-056; the
  expected set is asserted exactly). The `TODO.md` row says "every bundled `weights/` directory"; a card is the
  thing that cites, so the population is the cards, and a weights directory without a card has nothing to check. For each card, every cited file must exist in the card's own directory. A
  cited file is a back-ticked token that is a bare file name: `SHA256SUMS`, or a name with one of the extensions
  the bundles use (`.json`, `.npz`, `.md`). A name that a card cites from another place is on an explicit
  exemption list with its reason; the list is asserted to hold no entry that the scan does not hit. The test
  must fail on a card that cites a missing file (a detection case).
- (2) `.gitattributes` gets `silly_kicks/xsuccess/weights/metrics.json diff`, as lines 15-16 do for ghost-GK.
- The `TODO.md` row is removed in the same commit.

### C7 — the Task 27 Step 6 record

HFU spec `2026-11-04-hfu-reduce-reads-and-job-exit-design.md` §14.10a gets its result, from the cycle ledger
(`.superpowers/sdd/2026-11-05-hfu-reduce-reads-and-job-exit-plan/progress.md:459`): `check_params_ulp.py` over
`D:/Development/_staging/cycle1-a1a3b52`: 7 of 7 models within 3 ULP, `max_ulp=0`, exit 0; I3: `pool_coverage`
51,440 / 50,719, `keep_pool` false.

## 6. The parity launch (launch 8)

The launch is a parity proof. It is not a re-fit.

1. The cycle commit is pushed and CI is green (owner gates). The §4.4 job passes. The launch-7 run keys
   (`cycle1-a1a3b52-*`) stay in the bucket and the launch-7 staging folder stays on disk.
2. `preflight`, then `launch` of all 21 instances (owner go). `retrieve` and `verify` as in the howto.
3. **`campaign <file> compare --staging <new> --baseline <old>`** (new, read-only, local files). It reads the
   RETRIEVED artifact folders of both sides (`<staging>/<run_key>/<variant>/`, not `_final/`) and prints one line
   per instance. The rule is ONE definition in `scripts/hf_jobs/identity.py`, which `compare` extends:
   - a JSON file that is `metadata.json`, `metrics.json` or the file that THIS artifact's manifest `commit_field`
     names is compared after the provenance keys (`identity.PROVENANCE_KEYS`) are removed, recursively;
   - `SHA256SUMS` is compared on the lines of the files that are not normalized by the line above;
   - `MODEL_CARD.md` is compared after each 40-hex commit is replaced by a fixed token;
   - every other file is compared byte for byte;
   - the census is compared by `verify`'s rebaseline with `DriverSpec.tolerance`, as today, and `compare` prints
     that verdict.
   `artifact_size_bytes` joins `identity.PROVENANCE_KEYS` (owner 2026-11-10): it counts the bytes of files that
   hold provenance (M7), so it names the run, not the model. The payload files are still compared byte for byte.
   The only other reader of that list is `verify`'s census rebaseline (`adopt.py:435`, `:478`); the census records
   no such key. The trainers keep recording the value.
4. **Evidence before the launch.** The rule gives 16 of 21 on launch 6 against launch 7 (M7: the four F30
   `metrics.json` files and the census; the prototype and the rule as worded here give the same result). After
   the implementation the real `compare` command is run once on those two staging folders and its output is
   recorded in this section. This is recorded evidence, not a test: no test depends on a folder outside the repo,
   so no test skips. (Launch 7 against itself gives 21 of 21, which shows only that the command runs.)

   The output of `campaign scripts/hf_jobs/campaigns/cycle1.toml compare --staging
   D:\Development\_staging\cycle1-a1a3b52 --baseline D:\Development\_staging\cycle1-0552f58` after the
   implementation (exit 1: 16 identical, 4 different on `metrics.json` only, the census verdict ok):

   ```
   [compare] xsuccess-public/default IDENTICAL
   [compare] ghost_gk-owner/default IDENTICAL
   [compare] ghost_gk-owner/position_only IDENTICAL
   [compare] ghost_gk-owner/sweeper IDENTICAL
   [compare] ghost_gk-owner/sweeper_position_only IDENTICAL
   [compare] ghost_gk-owner/full IDENTICAL
   [compare] ghost_outfield-owner/default IDENTICAL
   [compare] ghost_outfield-owner/position_only IDENTICAL
   [compare] xshot_occurrence-public/default DIFFERENT: metrics.json: differs after removing the provenance keys
   [compare] xshot_occurrence-public/position_only DIFFERENT: metrics.json: differs after removing the provenance keys
   [compare] xshot_occurrence-sc_extended/default IDENTICAL
   [compare] xshot_occurrence-sc_extended/position_only IDENTICAL
   [compare] xcross_attempt-public/default DIFFERENT: metrics.json: differs after removing the provenance keys
   [compare] xcross_attempt-public/position_only DIFFERENT: metrics.json: differs after removing the provenance keys
   [compare] xcross_attempt-sc_extended/default IDENTICAL
   [compare] xcross_attempt-sc_extended/position_only IDENTICAL
   [compare] receiver_model-owner/default IDENTICAL
   [compare] win_probability-public/default IDENTICAL
   [compare] build_territory_ranking_census-public/default census verdict ok
   [compare] match_outcome_dependence-public/default IDENTICAL
   [compare] pass_completion-public/default IDENTICAL
   ```
5. **Pass:** the 20 artifacts are identical under the rule and the census rebaseline passes. Then nothing is
   adopted and nothing is published; the shipped artifacts keep `training_commit = a1a3b52f`. The launch record
   (the job count, the wall time, the `fit` stage time, the `compare` output, the §4.4 script sha256 and output)
   goes into this section in the evidence commit.
6. **Fail:** a difference is a defect of this cycle. It stops for the owner; it is never adopted.
7. The run keys: no deletion is decided here. After the evidence commit the owner rules which run keys `prune`
   deletes (the launch-8 keys are a proof record; the launch-7 keys are the run record of the shipped artifacts).

No tracked edit from `launch` until `compare` ends.

**Amended 2026-11-11 (§17.3):** step 3 reads the baseline side from its adopted copy (`_final/`) when it has
one. The launch record of step 5 is §17.

## 7. Tests (each seen to fail first)

- Part A: the `_fit_xgboost` oracle test with its boundary cases (A-R1); the reduce test through the store against
  the `_fit_work` oracle, once for each of the five shapes of A-R3 (A-R2, A-R3, A-R4); the xsuccess parity test
  on the unit grid (A-R5); the `units_per_job_by_stage` knob in `decide` (12 jobs of 3 for 36 units; the global
  value for every other stage; the round-robin path; an escalate job).
- Part B: the oracle equality test over the §4.3 inputs, the raising inputs included; `infer_ball_carrier`
  equality on the same inputs.
- Part C: the tests named in C1 to C6.
- §6: `compare` on toy staging folders: identical; a payload difference; a provenance-only difference; a card
  commit difference; a `SHA256SUMS` that lists `metadata.json`; the `match_outcome` shape (the commit inside
  `model.json`, the one `SHA256SUMS` line); an `artifact_size_bytes` difference (a provenance key); a non-provenance leaf
  difference (must fail).
- Local runs: the targeted tests and `tests/scripts/hf_jobs` only. CI runs the full suite and full pyright.

## 8. Docs that travel with the commit

`ADR-128` (one amendment: the scope decisions of §1.3 with M1 to M8, Parts A and C, the launch-8 rule),
`docs/howto/hf-jobs-training.md` (check 7, `prune`, `compare`, the STUCK section, the lost-result report, the
`post_verify` rule, `units_per_job_by_stage`), `docs/howto/compute-pipelines.md` (the xsuccess `fit` as the example
of units in place of an InnerPool), `docs/context/tracking-features.md` (the pre-index), `CHANGELOG.md`, `TODO.md`
(the removed row; the "Current" line), the `hf-campaign` skill text if it names the commands, the C4 model if a
container changes (none expected). The version number is set at commit preparation.

## 9. Order and definition of done

1. A/B approval of this spec; then the plan and its A/B review.
2. Parts B and C test-first (independent of each other), then Part A.
3. A/B implementation review; `/final-review`; the cycle commit and push (owner yes); CI green.
4. The §4.4 job (owner go). Launch 8 and `compare` (§6, owner go).
5. The evidence commit (owner yes). PR, merge and tag, each on its own yes.

A defect found in step 4 is fixed by an amend and a force-push, each on its own owner yes, and the failed step
runs again.

Done = CI green on the cycle commit, the §4.4 job passes, launch 8 passes §6 for all 21 instances, and both
reviewers approve each stage.

## 10. Author decisions for the owner to check

These are the author's calls. None is an owner ruling yet.

| Id | Call |
|---|---|
| D1 | No stored feature matrix and no `prep` stage: each unit reads the 75 MB `extract` output and builds its own rows. (Revision 1 stored a 2.2 GB matrix; round 1 showed a fold-index file near 2 GB and one copy per worker.) |
| D2 | The calibration, the metrics and the bundle are built in the `fit` reduce, so the output stage keeps the name `fit`. |
| D3 | No memory probe: the 7.26 GiB of a calibrated fit is the bound of a unit (the hint is 8 GiB), and the new knob `units_per_job_by_stage` runs all 36 fits at once in 12 jobs of 3. |
| D4 | The storage bound is 100 GB with 20 GB headroom, as campaign-file values. |
| D5 | `prune` works per commit; the approval flag names each commit. |
| D6 | The `FileSet` member check runs in `plan()` only; a running driver reruns a lost unit within its attempts and then reports it. |
| D7 | `compare` is a new read-only campaign command; its rule lives in `identity.py`. |
| D8 | C5 is a message and a howto section, with no behaviour change. |
| D9 | RULED (owner 2026-11-10): `artifact_size_bytes` is a provenance key in `identity.PROVENANCE_KEYS`. |
| D10 | RULED (owner 2026-11-10): Part B gets a one-time production check on three real matches (§4.4); it needs an owner go when it runs. |
| D11 | C4 does not cover a member that is present and corrupt; a downstream unit fails on it, as today. |

## 11. Not verified

- The peak RSS of one fit unit (§3.4 bounds it; launch 8 records it).
- Whether the explicit calibration loop equals `cross_val_predict` byte for byte (A-R1 decides).
- The speed of the new pre-index (§4.4 and the plan measure it).
- That numpy's `hypot` and `arctan2` give the same bits for a row whatever its position in the array on the HF
  image. Part A no longer depends on it: each unit computes the features of its own rows, as today.
- What HF counts against the org storage limit (C3). The limit itself is owner-verified: 100 GB private.
- The storage stop length and the HTTP 429 count of launch 7 (handoff text; no log read).

## 12. Round 1 answers

Reports: `D:\Development\_reviews\2026-11-10-hfu-fit-units-spec-A.md`, `…-spec-B.md`. A and B did not disagree.

| Finding | Answer |
|---|---|
| A-01, B-01 the compare rule | Fixed: §6 step 3 (the `SHA256SUMS` rule, the per-artifact `commit_field` file, `artifact_size_bytes`), step 4 (the rule run on launches 6 and 7), §7, M7. The package-version clause is removed (A-10). |
| A-02, B-02 the row space | Fixed: §3.2 (the pooled row space; the reduce reads the pooled rows), A-R2, §3.5. |
| A-03, B-07 result forms, chirality, small shapes | Fixed: §3.2 unit table, the absent units, `per_type_logistic`, fewer than 2 matches; A-R1 boundary cases, A-R3, A-R4. |
| A-04, B-08 `plan.npz`, the per-worker copy | Removed with the `prep` stage (D1). |
| A-05, B-10 the probe and the commit count | Removed: no probe (D3, §3.4); one code commit; the amend path is named in §9. |
| A-06, B-11 C4 in a running driver | Fixed: C4 (the in-run behaviour, the report text, `ready`, the valid-`_output` case, the corrupt member, the `decide` test, the count rule, `datasets.py`). |
| A-07, B-03 Part B semantics and equality | Fixed: §4.2 (what is rewritten), §4.3 (B-S1 to B-S9, the equality rule, the raising inputs, the adversarial list). |
| B-04 Part B at production scale | Fixed: §4.4. |
| A-08, B-06, B-14 to B-16 `prune` and storage | Fixed: C3; §6 step 7 decides no deletion. |
| A-09, B-05 scope record | Fixed: §1.3 carries a row for every handoff item, item 2 whole and item 9 whole, with the owner's words; C7's ruling is named; memory `project_hfu_next_cycle_scope.md` holds them. |
| B-09 C6 | Fixed: C6. |
| A-11 to A-21, B-12, B-13, B-17 to B-20 | Applied: A-R6 names `metrics.json`; the staging layer; the C2 gate; §3.5 files; the heavy reduce; §1.4; the caller list; the C1 bound; the M-row wording; one rule in `identity.py`; the `FutureWarning`; the boundary cases. B-12 (A-R1 and A-R2 in the probe) falls with the probe; A-R1 keeps its toy gate and §6 is its production witness. B-19 (the `final-*` units in one job) falls with 12 jobs of 3. |

## 13. Round 2 answers (revision 3)

Reports: `D:\Development\_reviews\2026-11-10-hfu-fit-units-spec-r2-A.md`, `…-spec-r2-B.md`. Both APPROVE.

| Finding | Answer |
|---|---|
| A-22 the small shapes | Fixed: A-R3 and §7 (the reduce test through the store for each of the five shapes). |
| A-23 a skipping test | Fixed: §6 step 4 (recorded evidence; no test depends on a folder outside the repo). |
| A-24 the §4.4 script | Fixed: §4.4 (the script goes to both reviewers; sha256 and output in the evidence commit; the non-vacuity asserts). |
| A-25 B-S1 | Fixed: both `is_ball` dtypes. |
| A-26 the C6 population | Fixed: C6 states the difference from the `TODO.md` row. |
| A-27, B-24 the §6 evidence | Fixed: §6 step 4, M7. |
| A-28, B-22 the reduce memory | Fixed: §3.4. |
| A-29 the knob's reach | Fixed: §3.4, §7. |
| B-21 the cap margin | Fixed: 8 GiB (§3.4). |
| B-23 the A-R3 reference | Fixed: the `_fit_work` serial path as the oracle. |
| B-25 §10 | Fixed: D11; "a later cycle can widen" is removed. |
| B-26, B-27 references | Fixed: §4.1, `cycle1.toml:32-33`, `driver.py:655-658`, the cap value. |
| Owner rulings 2026-11-10 | D9 (`artifact_size_bytes` is a provenance key) and D10 (a one-time check on three matches, not 179). |

## 14. Build notes (what the implementation found)

| Id | Note |
|---|---|
| N1 | **A corpus of successes only raises.** In the 4.140.0 fit, xgboost refuses a label set of {1} (`ValueError: Invalid classes inferred`). An all-FAIL corpus fits without calibration. So the "one class" shape of A-R3 is the all-fail corpus; the all-success corpus is tested as "raises the same type as before". |
| N2 | **Two more private steps than §3.2 names:** `_calibration_oof` (the placement of the calibration folds) and `_write_metadata_and_sums` (the two file writes of `save`). `save` uses `Path.write_text`, which writes `\r\n` on Windows, so the reduce must write the two files with the same code to give the same bytes on each platform. |
| N3 | **The model oracle also holds the old `save` and `_sha`** (`_model.py:141-352`), so no oracle path runs new file-writing code. |
| N4 | **The `FutureWarning` of §4.5 came from three `groupby` calls** of `_pre_index_frames` (the ball aggregation, the player groups, the segments). The rewrite removed the player-groups one only. Both reviewers reported the conflict of §4.2 with §4.5. **Owner ruling 2026-11-10:** the two other calls state `observed=True`; no warning is left (`test_category_frame_keys_give_no_future_warning`). A read-only probe before the change: equal output on all 25 named inputs for `observed=True` and for `observed=False`. |
| N5 | **The ADR-073 growth guard of the pre-index is a `SCALE_GUARDED` entry** (`tests/test_scale_guards.py::test__pre_index_frames_is_subquadratic`), as plan Task 1 Step 7 says. The first build kept it beside the equality tests with the reason "the registry is pinned to the callers of `group_rows`". That reason was wrong (reviewer A): the registry gate is a superset check, and 4 of its entries are not `group_rows` callers. The structural test (no per-frame object) stays beside the equality tests, because it runs the oracle. |
| N6 | **The frame-lookup dict of `_pre_index_frames` (former lines 94-97) is removed** with the placement loop, its only reader. It is outside the line range of §4.2. **Owner ruling 2026-11-10:** approved (§4.2). |
| N7 | **No bundled card cites a missing file** (8 cards), so the exemption list of C6 is empty. |
| N8 | **`datasets.fetch` (C4, the download plan) runs for no locked dataset:** none has `access = "download"`. |
| N9 | **Timing on this PC** (the synthetic 20,000-frame match, 460,000 rows, best of 3): the oracle 4.177 s, the new function 0.229 s. |
| N10 | **`assert_pre_index_equal` compares float arrays by bits** (reviewer A-09): stricter than the §4.3 rule. The Task 14 job uses the same function, so the three real matches are compared by bits too. |
| N11 | **`post_verify` empties `<staging>/_post_verify/<repo>/` before the download** (reviewer A-03): a file that an earlier publish left there cannot pass for a file of the Hub. |
| N12 | **The `--plan` call of each instance in `retrieve` holds one slot of the read bound** (reviewer A-08; C1 says EVERY store read). `plan()` reads one request at a time, so one slot is its exact count. |
| N13 | **Check 7 fails closed on a listing entry without `private`** (reviewer B-05), and its record name is `storage`, as C3 says (the function is `check_storage`). Live read-only check 2026-11-10: the default `list_datasets` and `list_models` answers carry `private` (6 datasets; 20 models, 10 private). |
| N14 | **`prune` reads a 403 from the HTTP status code or from the text "storage limit reached"** (the phrase that `failures.classify` matches), not from the digits 403 anywhere in an error text (reviewers B-06, A-07). A refusal ends `campaign <file> prune` with one line (`prune refused: ...`), not a traceback. |
| N15 | **The shape tests of the `fit` units use study parameters on the main corpus** (reviewer B-07; production always has them), and the main corpus has seven matches so that the two row spaces give different fold PARTITIONS (reviewer A-01). A permanent test takes the folds from the finite rows and must fail the byte comparison. |
| N16 | **`tests/scripts/hf_jobs/test_job_liveness.py::test_alive_lines_name_the_stuck_function` is made deterministic** (owner ruling 2026-11-10; the test is not part of the three parts): two tests (the beats; the dump), the child sets its own deadline after a warm numba import, and the dump assert reads the dump's own lines. |
| N17 | **The deadline dump of `scripts/hf_jobs/liveness.py` is printed by a Python thread, not by the `faulthandler` watchdog** (owner approval 2026-11-10, after two read-only HF measurements that he asked for). Root cause of the failed test: the watchdog dump reads a running thread's frames without the GIL. Job `6af366809737d9a9e0914f3a`: a dump inside the numba import ended 62 of 206 processes (SIGSEGV); the production path with its 1 s floor: 0 of 204. Job `6af3700bab34849891acb4a8` (40 processes per workload, a dump every 0.05 s for 3 s, where production prints one dump per process; the processes ended by SIGSEGV, one by SIGBUS): 18 of 40 during an xgboost fit, 38 of 40 during pandas work, 40 of 40 during `infer_ball_carrier`, 40 of 40 during a numba compile, 36 of 40 during the numba import; 0 of 40 while the main thread slept, ran pure Python calls or a compiled numba kernel. Launch 7 printed no dump. The change: `stack_dump` and `_dump_later` (a daemon thread `sk-dump`), the same dump times (`dump_delay_s`), the `faulthandler` line format; `_say` writes a record in one write (the first proof job, `6af377e89737d9a9e0917dd3`, showed a dump header on the end of another line and stopped on the probe's own parser). **Proof:** job `6af378cb9737d9a9e0917f86`, the same 8 workloads with the new `stack_dump` (embedded verbatim, sha256 `e93b40e1…fa949`) printed every 0.05 s: 0 of 320 processes ended by a signal, about 14,500 dumps (sleep 2,360; numba import 187; pure Python 2,001; numba compile 2,211; numba kernel 1,453; xgboost fit 2,497; pandas 2,372; `infer_ball_carrier` 1,419). The lower counts under a compiled kernel show the cost: the dump waits while native code holds the GIL. Cost: a dump waits while a C extension holds the GIL in one long call. The scripts and logs: `D:\Temp\hfu-next\liveness_hf\`. |

## 15. Implementation review round 1 answers

Reports: `D:\Development\_reviews\2026-11-10-hfu-fit-units-impl-A.md` (APPROVE; 0 blocking, 3 should fix,
6 consider) and `…-impl-B.md` (REQUEST CHANGES, narrow; 0 blocking, 2 should fix, 7 consider). Neither found a
code defect. A and B did not disagree. Every fix was written test-first (the test failed, then passed).

| Finding | Answer |
|---|---|
| A-01 the A-R2 gate cannot fail (fold numbers, the same partition) | Fixed: N15. On the six-match corpus the new partition assert and the new fault test both failed; on the seven-match corpus both pass. |
| A-02, B-03 the guard is not in `SCALE_GUARDED`; N5's reason | Fixed: N5 (registered as planned; the wrong reason is corrected). |
| A-03 `post_verify` can load a stale local file | Fixed: N11 (`test_post_verify_does_not_load_a_stale_local_file`). |
| B-01, A-05 the `FutureWarning` stays; §4.2 against §4.5 | Fixed by owner ruling: N4, §4.2, §4.5. |
| B-02 the `TODO.md` "Current" line and the `CHANGELOG.md` heading | Fixed at commit preparation: the owner gave 4.140.1 (2026-11-10; PR-S204). `silly_kicks/_version.py`, the `CHANGELOG.md` heading, the `TODO.md` "Current" line and `uv.lock` hold it. |
| B-03 N6 outside the line range | Owner ruling: approved (N6, §4.2). |
| A-04, B-04 the check name; "six" in the docstring | Fixed: N13; the module docstring says seven. |
| B-05 a repo without `private` is not counted | Fixed: N13. |
| B-06, A-07 the 403 rule; a refusal gives a traceback | Fixed: N14. |
| B-07 no shape test has study parameters | Fixed: N15. |
| A-06, B-08 the ADR amendment has no table | Fixed: the amendment holds the M1 to M8 table and the scope table of §1.3, with D1-IMPL-09, -11, -12. |
| A-08 the `--plan` reads of `retrieve` are outside the bound | Fixed: N12 (`test_retrieve_runs_the_plan_reads_under_the_bound`). |
| A-09 `-0.0` equals `0.0` in the equality rule | Fixed: N10. |
| B-09 `compare` matches no `-r<N>` run key | Owner ruling 2026-11-10: left as it is. No replica is planned, and the result is loud (a difference, exit 1). |
| The author's failed test (`test_job_liveness.py`) | Owner ruling 2026-11-10: fixed in this cycle, root cause first (N16). The root cause is a defect of the job watch itself; the owner approved its fix in this cycle after two HF measurements (N17, ADR-128 item 47). |
| N1, N2, N3, N7, N8; the `pyproject.toml` `N803` ignore | Both reviewers checked them and found them right; shown to the owner with the rulings above. |

## 16. Implementation review rounds 2 and 3

Reports: `D:\Development\_reviews\2026-11-10-hfu-fit-units-impl-r2-{A,B}.md`, `…-impl-r3-{A,B}.md`,
`…-impl-r3b-B.md`, `…-impl-r4-{A,B}.md`.

| Round | A | B | Result |
|---|---|---|---|
| 2 (the round-1 fixes) | APPROVE; all nine findings resolved | APPROVE; one item open by plan order (the version) | No new finding. |
| 3 (narrow: the deadline dump of N17 and the version) | APPROVE; 1 consider | REQUEST CHANGES, narrow; 1 should fix, 1 consider | See the rows below. |
| 3b (B confirms its finding) | not asked | APPROVE | Nothing is open. |
| 4 (narrow: the plan-timeout dump, see below) | APPROVE; 1 consider (this section's sentence on the amend) | APPROVE | Nothing is open. |

| Round-3 finding | Answer |
|---|---|
| B HFU2-IMPL-10: no test bound the delay that `start_job_watch` gives the dump thread (a mutant with `delay = 1.0` stayed green) | Fixed: `test_the_dump_thread_gets_the_delay_of_its_role` (four roles; a far deadline and one inside the margin). Against that mutant: 4 failed. |
| A HFU2-IMPL-10: the counts of N17 stood without the measurement condition | Fixed: the `liveness.py` docstring, `CHANGELOG.md`, ADR-128 item 47 and N17 state it (a dump every 0.05 s for 3 s per process; production prints one). |
| B HFU2-IMPL-11: 35 SIGSEGV and 1 SIGBUS, not 36 SIGSEGV | Fixed: "ended by a signal (SIGSEGV; one SIGBUS)" in the same four texts. |
| B round 1 HFU2-IMPL-02 (the version) | Closed: 4.140.1 (owner 2026-11-10). |

The plan-timeout dump: `scripts/_stages.py` armed the same `faulthandler` watchdog 30 s before the controller ends
a hung `--plan` subprocess. The owner asked whether to fix it and approved the fix in this cycle (2026-11-10,
after the cycle commit; the commit was amended on his yes). `arm_plan_deadline` starts the dump thread of the job
watch (`liveness.start_dump_thread`, role `plan`); `campaign._plan_timeout_detail` reads the new header; the
helper that moved the watchdog's main-thread block to the front is removed, because the new dump lists the main
thread first. No `faulthandler` call is left in the campaign code.

## 17. Launch 8: the record, three findings, three owner rulings (2026-11-11)

Launch 8 ran at `abcfda86` (owner go 2026-11-10). This section is the record of §6 step 5. Every number of §17.1,
§17.2 and §17.4 was measured before any tracked file changed. Scratch scripts and logs: `D:\Temp\hfu-next\launch8\`.

### 17.1 The launch record

| Item | Measured |
|---|---|
| Preflight | 7 of 7 checks ok. Storage before the launch: 49.93 GB used (the scratch bucket 27.74 GB, the private datasets 22.06 GB, the canary bucket 0.12 GB). |
| Driver | 2026-11-10T22:29:33Z to 2026-11-11T02:12:10Z: 3 h 42 min 37 s. Exit 0. 208 steps; no FATAL, no STUCK, no failed `--plan`, no rate-limited step. |
| Jobs | 397: 394 COMPLETED, 3 ERROR. The 3 are `mount_auth_flake` (pass_completion `fit`, ghost_gk sweeper `importance`, ghost_gk sweeper_position_only `cache`); the driver relaunched each (budget 12 per dataset). |
| Instances | 21 of 21 reduced. `retrieve` 21 of 21. `verify` 21 of 21 ok. |
| xsuccess `refs` + `extract` | From the driver start to the `hpo` launch: 40 min 28 s (37 `extract` jobs). Launch 7: 25 min (M1). Code not changed; cause not investigated. |
| xsuccess `hpo` | One unit: 9,654 s (160.9 min) for 30 trials, mean 322 s per trial (early trials about 285 s, late trials 367 to 670 s). Launch 7: 141.2 min. Code not changed; cause not investigated. |
| xsuccess `fit` | 36 units in 12 jobs, every unit done on attempt 1. From the stage launch (01:52:34Z) to the reduced stage (the driver end): 19 min 36 s. Launch 7: 63 min. Unit wall time 654.9 / 805.6 / 994.8 s (min / median / max); first unit start to last unit end 1,004.5 s; the reduce 65.4 s. |
| `fit` peak RSS | 6.02 / 6.12 / 7.26 GiB (min / median / max, `rss_scope` unit). The hint is 8 GiB, so 3 units per job is right. |

Against §1.4: the `fit` stage went from 63 min to 19 min 36 s, not to about 12 min. One unit took 655 to 995 s
where §3.4 expected about 430 s (three units run at once in one job; the cause of the difference was not
investigated). The complete run took 3 h 43 min, not about 3 h: `refs` + `extract` and `hpo` together took 35 min
more than at launch 7 with unchanged code. The long pole is still xsuccess: 40 + 161 + 20 min.

### 17.2 The production check of §4.4

One read-only job, `6af389acab34849891acef67`, at `abcfda86`, `cpu-upgrade`, COMPLETED in 542 s, `PRE_INDEX_OK 3`.
The in-job script: sha256 `3af84fe529da387275a44c414f31b4c6c2bb7cb555cee73a91665369092fd3b4`; the launcher: sha256
`0ff3b50d0309f110a2cf2bb9002f7cab332463c5a7881f6650fbcef8bb384cf0` (both outside the repo, both read by A and B).

| Match | Rows | Frames | Id dtypes | Oracle | New | Speed-up | Output | Carrier |
|---|---|---|---|---|---|---|---|---|
| `gradientsports__10502` | 4,000,957 | 175,969 | `player_id` Int64, `team_id` float64 | 62.52 s | 1.80 s | 34.8x | equal by bits | equal |
| `skillcorner__1021404` | 1,051,215 | 45,705 | `player_id` object, `team_id` category | 12.86 s | 0.61 s | 21.0x | equal by bits | equal |
| `idsse__DFL-MAT-J03WMX` | 3,357,241 | 145,967 | `player_id` object, `team_id` category | 40.30 s | 1.75 s | 23.0x | equal by bits | equal |

### 17.3 `compare`: the result, a finding, the baseline rule

**The result with the rule of §6 step 3 as written** (`--baseline D:\Development\_staging\cycle1-a1a3b52`, the
retrieved files): exit 1. 16 trainer instances IDENTICAL (xsuccess among them), the census verdict ok, 4 DIFFERENT:
`xcross_attempt` `public` and `sc_extended`, both variants, each with `metrics.json: differs after removing the
provenance keys`. `model.json`, `metadata.json` and `SHA256SUMS` of the four are equal.

**The cause.** Six leaves of each `metrics.json` differ: `probe_sample_matches[i][1]` and the keys of
`probe_sample_in_training_folds`. Launch 7 wrote the staged names `gradientsports-10502` and `gradientsports-10503`;
launch 8 wrote the canonical ids `10502` and `10503`. That is the trainer fix of HFU spec §18.6 (owner 2026-11-10:
"fix the trainer, and canonicalize at adoption; no rerun"). It entered the code with `53e529b7` (4.140.0), after
launch 7 ran at `a1a3b52f`, so launch 8 is its first run on HF. This cycle did not change that trainer
(`git diff bb0c601c..abcfda86` touches neither `scripts/train_xcross_attempt.py` nor `scripts/_corpus_identity.py`).
So the difference is not a defect of this cycle (§6 step 6). **The defect was in §6 step 3 (the author's):** it
named the retrieved files of launch 7 as the baseline. The shipped form of launch 7 is its adopted copy.

**Proof that launch 8 equals the shipped form.**

- The tool of `abcfda86`, unchanged, with `--baseline D:\Development\_staging\cycle1-a1a3b52\_final` (the adopted
  copy, which has the same folder layout): exit 0, 20 of 20 IDENTICAL, the census verdict ok.
- For each of the four xCross instances: `canonical_probe_ids(launch-7 retrieved metrics)` equals the launch-8
  retrieved metrics without the provenance keys, and `canonical_probe_ids` changes nothing in the launch-8 metrics.
- The adopted copy is the shipped form: for xCross `public`, its `model.json`, `metadata.json` and `SHA256SUMS`
  equal the wheel bundle byte for byte; its `metrics.json` differs from the bundle only by the public-id
  rehydration that `adopt` does (`corpus_match_ids` in place of the digest and the count).

**The owner's ruling (2026-11-11, "1: fix here"): the tool gets the right baseline, so that this cannot return.**
Launch 8 adopts nothing, so launch 7 stays the shipped launch and the baseline of every later parity proof.

- `adopt.compare_staging` reads the baseline side from its adopted copy, `<baseline>/_final/<run_key>/<variant>/`,
  when the baseline has one for the instance, and from its retrieved files otherwise. `adopt._final_dir` is the
  one path helper; `_finalize` writes that folder.
- The new side is always its retrieved files. A job must write the shipped form itself, so a trainer that goes
  back to another form shows as DIFFERENT.
- Each row has `baseline`: `adopted` or `retrieved`. The line reads `IDENTICAL (baseline: adopted copy)` or
  `DIFFERENT (baseline: retrieved files): ...`. An instance in only one folder names no form.

The output of `campaign scripts/hf_jobs/campaigns/cycle1.toml compare --staging
D:\Development\_staging\cycle1-abcfda8 --baseline D:\Development\_staging\cycle1-a1a3b52` with the changed tool
(exit 0):

```
[compare] xsuccess-public/default IDENTICAL (baseline: adopted copy)
[compare] ghost_gk-owner/default IDENTICAL (baseline: adopted copy)
[compare] ghost_gk-owner/position_only IDENTICAL (baseline: adopted copy)
[compare] ghost_gk-owner/sweeper IDENTICAL (baseline: adopted copy)
[compare] ghost_gk-owner/sweeper_position_only IDENTICAL (baseline: adopted copy)
[compare] ghost_gk-owner/full IDENTICAL (baseline: adopted copy)
[compare] ghost_outfield-owner/default IDENTICAL (baseline: adopted copy)
[compare] ghost_outfield-owner/position_only IDENTICAL (baseline: adopted copy)
[compare] xshot_occurrence-public/default IDENTICAL (baseline: adopted copy)
[compare] xshot_occurrence-public/position_only IDENTICAL (baseline: adopted copy)
[compare] xshot_occurrence-sc_extended/default IDENTICAL (baseline: adopted copy)
[compare] xshot_occurrence-sc_extended/position_only IDENTICAL (baseline: adopted copy)
[compare] xcross_attempt-public/default IDENTICAL (baseline: adopted copy)
[compare] xcross_attempt-public/position_only IDENTICAL (baseline: adopted copy)
[compare] xcross_attempt-sc_extended/default IDENTICAL (baseline: adopted copy)
[compare] xcross_attempt-sc_extended/position_only IDENTICAL (baseline: adopted copy)
[compare] receiver_model-owner/default IDENTICAL (baseline: adopted copy)
[compare] win_probability-public/default IDENTICAL (baseline: adopted copy)
[compare] build_territory_ranking_census-public/default census verdict ok
[compare] match_outcome_dependence-public/default IDENTICAL (baseline: adopted copy)
[compare] pass_completion-public/default IDENTICAL (baseline: adopted copy)
```

**Pass (§6 step 5):** nothing is adopted and nothing is published. The shipped artifacts keep
`training_commit = a1a3b52f`.

Tests (each seen to fail first: 6 failed): the adopted copy is read on the baseline side, and an instance without
one is read from its retrieved files; the adopted copy is the authority when the retrieved files of the baseline
equal the new launch and the adopted copy does not; the new side never reads its own adopted copy; `_finalize`
writes the folder that `compare` reads; the three line forms. Mutation check: 5 of 5 mutants caught.

### 17.4 The plan request counts, a second finding, the request count in the step line

One `--plan` of four finished instances, through a counting hook on the Hub client (read-only):

| Instance | Reference (4.140.0) | Bound of C4 as written | Measured | Listing pages | Other requests |
|---|---|---|---|---|---|
| win_probability | 16 | 18 | 20 | 12 | 8: `refs` `_output.json` and `.sha256`; the `fit` markers `_output` and `only` |
| xsuccess | 16 | 19 | 21 | 13 | 8: `refs` `_output.json` and `.sha256`; the `fit` markers `_output` and `final-main` |
| census | 17 | 19 | 21 | 13 | 8: `refs` `_output.json` and `.sha256`; the `report` markers `_output` and `only` |
| ghost_gk/default | 4 | 8 | 12 | 4 | 8: the `fit` and `assemble` markers, `_output` and `only` each |

- **The finding (the author's error in C4):** one store read is TWO HTTP requests with huggingface_hub 2.2.0,
  `POST paths-info` and `HEAD resolve` (the bytes come over xet). C4 counted a marker read as one request. So
  every measured count is above the bound as written, by the number of `FileSet` results of the instance.
- **The design holds exactly:** one marker read per `FileSet` result and no extra listing. Measured = reference
  + 2 x `FileSet` results, plus one listing page for xsuccess (its run key holds 12,105 files, 11,973 at launch 7;
  a page is 1,000 files).
- **Effect in launch 8:** none seen. 208 steps, no rate-limited step, no failed `--plan`.

**The owner's ruling (2026-11-11, "1: Do it all"), three parts:**

1. **The bound is in store operations**, which the code controls: per plan, one recursive listing and one marker
   read per `FileSet` result. `test_the_member_check_adds_one_marker_read_per_fileset_result_and_no_listing`
   binds it. The HTTP counts above are a measured fact of one library version, not a bound.
2. **The driver measures its own requests**, so that no scratch script is needed again (the 4.140.0 script was not
   kept, and the first rewrite of it in this cycle counted 0, because the campaign code installs its own client).
   `hub_client.count_request` is a request hook of the bounded client: it counts every request of the process (a
   retry and a listing page are requests). A `--plan` process of a bucket store prints `[plan] hub requests <n>`
   on stderr when it ends; `campaign.read_plan` adds that count to the controller's tally
   (`hub_client.add_plan_requests`); a plan that fails reports its count too, and a plan that timed out reports
   none (amended in review round 5, §18). The step line prints
   `hub requests <total> (plans <p>)`: the controller's own requests plus those of its plan processes. A
   rate-limited step prints `rate-limited after <s>s (hub requests <total>); backing off`. The plan JSON does not
   change.
3. **No request is removed now.** There is no measured need (part 2 gives the measure for the next launch), and
   the large term is the listing, not the markers.

Tests (each seen to fail first: 11 failed): the client carries the counting hook, and 8 threads x 500 requests
count 4,000; the plan line round trip and the last-line rule; a process without the client prints no line;
`read_plan` adds the count and leaves the plan JSON; a `--plan` of a local store prints no line and one with the
client prints it after the JSON; the step line with the controller's and the plans' requests over two steps; the
rate-limited line. Mutation check: 10 of 10 mutants caught. The first form of the counting test let the mutant
"the hook counts without the lock" pass (a read, a short loop, a write); the test now sets the smallest thread
switch interval and catches it. A bare `+= 1` without the lock has no switch point between its read and its write
on CPython 3.12, so no test tells it from the locked form; the lock is the stated guarantee.

### 17.5 A test order dependence that the final check found, the third owner ruling

The final check of the second commit ran `tests/scripts/hf_jobs` and four more test files in one process: 640
passed, 1 skipped, 1 failed. The failure was `tests/scripts/test_unit_log_lines.py::test_job_summary_line[ok-0]`,
with `LostWriteError`. It is not a finding of launch 8.

- **Cause:** `scripts/_write_confirm.py` keeps one journal of written files for each process. A job process
  confirms its writes when it ends, and then it exits. A test process does not exit between tests. Two tests of
  `tests/scripts/hf_jobs/test_executor.py`
  (`test_a_deferred_memory_unit_neither_fails_a_strict_stage_nor_lets_it_reduce`,
  `test_the_stage_cli_escalate_flag_runs_a_deferred_unit`) call `record_memory_failure` in the test process, and
  a worker subprocess then deletes that marker when the escalated unit is done. The entry stays in the journal of
  the test process. `test_job_summary_line[ok-0]` runs a real job exit (`_stages.main`), which looks for every
  journaled file.
- **No product defect found (read, not tested):** in a job, the process that records a memory failure keeps the
  claim of that unit. `executor._dispatch` frees a claim only for a `pending` unit, and a recorded unit is
  `failed`, so no worker of that job runs the unit again and deletes the marker.
- **Not from the second commit:** the three tests and `_write_confirm.py` are equal to `abcfda86`. The author did
  not run the failure at a clean copy of `abcfda86`; both reviewers did in round 5, and it fails there (§18).
- **CI did not see it:** the 12-group split puts the two executor tests in group 3 and `test_job_summary_line` in
  group 7 (a collect-only listing of the tree with the new tests). A run in one process sees it, and that
  includes the documented local command `python -m pytest tests/ -m "not e2e"`.

**The owner's ruling (2026-11-11, "1"): fix here.** `tests/scripts/conftest.py` has one autouse fixture,
`_empty_write_journal`. It empties the journal before and after every test under `tests/scripts/`.
`tests/scripts/test_write_confirm.py` had the same fixture for its own tests only; that fixture moved, so there is
one.

Test (seen to fail first): `test_a_journal_entry_does_not_reach_the_next_test`, two cases. Each case expects an
empty journal at its start and leaves one entry on purpose, so the case that runs second fails in either order.
Without the fixture: 6 failed and 10 passed in `test_write_confirm.py`, and the three tests above gave 1 failed
and 4 passed. With the fixture: 16 passed, and 5 passed. Limit: one of the two clears alone passes this test (the
clear after a test does the work of the clear before the next test). Both stay, as in the former fixture: the
clear before a test also removes an entry that a test outside `tests/scripts/` left.

## 18. Implementation review round 5 (the second commit) and three more owner rulings

Reports: `D:\Development\_reviews\2026-11-10-hfu-fit-units-impl-r5-{A,B}.md`. Both reviewed the tree with the
`git diff HEAD` sha256 `406be5c5…c0e2` (15 files). A: APPROVE, 2 should fix. B: APPROVE, 1 consider. Both
reproduced the real `compare` (the tool of `abcfda86` exit 1; the changed tool exit 0, 20 of 20). Both ran the
failure of §17.5 at a clean `abcfda86`: it fails there (A: 1 failed, 7 passed; B: 1 failed, 4 passed), so it is
not from the second commit. Both read `executor._dispatch` and agree with "no product defect"; no test binds it.
A's revert proof: 26 failed, 174 passed. B's mutation run: 16 of 16 caught.

| Finding | Answer |
|---|---|
| A HFU2-IMPL-12 (should fix): `test_plan_prints_its_hub_request_count_only_with_a_hub_client` needed `hub_client._INSTALLED` to be `False`, and `test_package_main_routes_campaign_to_the_driver` leaves it `True` | Fixed: the test sets both client states itself. A's command gave 1 failed, 7 passed; it gives 8 passed. The cause is fixed too (owner ruling 5). |
| A HFU2-IMPL-13 (should fix): the step-line count left out a `--plan` that failed or timed out, and only this spec said so | Fixed by owner ruling 4. |
| B HFU2-IMPL-12 (consider): the how-to, ADR-128 item 48 and the `compare_staging` docstring said that `_final/` holds "the files that `adopt` bundled" | Fixed: the three texts say "finalized". The how-to and the ADR name the difference that §17.3 states (xCross `public`: the bundled `metrics.json` holds `corpus_match_ids`, the adopted copy their digest and their count). |

**The owner's rulings (2026-11-11, "Gold standard for all three"):**

4. **A plan that fails reports its count.** `_stages.main` prints `[plan] hub requests <n>` in a `finally` that
   covers the store open and the plan read, and `campaign.read_plan` adds the count before it raises. So a 429
   inside a plan is in the rate-limited line of the step. A plan that timed out was killed and reports none: the
   count of that step is a lower bound, and the how-to, `CHANGELOG.md`, ADR-128 item 49 and the docstrings say
   so. No count comes from the stack dump of a timeout (it would still be a lower bound, and it would tie the
   dump code to the request counter).
5. **The test that left the Hub client installed is fixed at its source.**
   `test_package_main_routes_campaign_to_the_driver` (in `abcfda86`) called the real `install_timeouts`, which
   replaces the client factory of the process. The test now replaces that function, asserts one call, and
   asserts that `hub_client._INSTALLED` is as it found it. No directory-wide fixture: this test is the only one
   that left the state.
6. **The four timings of §17.1 get one `TODO.md` row** ("Find the cause of four launch-8 timings before the next
   HF launch"). The `fit` unit time comes first, because `units_per_job_by_stage` rests on the 430 s estimate.

Tests (each seen to fail first, 5 failed): a `--plan` that fails in the store open or in the plan read prints its
count last on stderr and nothing on stdout; `read_plan` adds the count of a plan with a non-zero exit and then
raises; a step whose real `read_plan` gets a 429 from the plan process prints `hub requests 9` in its
rate-limited line; the package-main test with the `_INSTALLED` assertion and without the replacement. One test
passed at once, because it states the limit and not a change: `read_plan` adds nothing for a plan that timed out.
After the change: `test_unit_log_lines.py` 9 passed, `test_hub_client.py` 17 passed, the step selection 5 passed.
No separate mutation run: the three reverts (the line outside the `finally`, the parse after the raise, the test
without the replacement) are the state of the failing run.

**Round 6 (narrow: the answers above).** Reports: `…-impl-r6-{A,B}.md`, on the tree with the `git diff HEAD`
sha256 `4a1edaa6…69f2` (16 files). A: APPROVE, no finding; its two findings RESOLVED; its revert proof gave 4
failed, 23 passed. B: APPROVE, no finding; its finding RESOLVED; four mutants caught. Only this paragraph changed
after round 6. On the same tree the author ran `test_stage_reads.py` (21 passed) and `test_stage_contract.py`
(35 passed), because `scripts/_stages.py` changed and neither reviewer ran them at this state. Not run at this
state: all of `tests/scripts/hf_jobs` in one process and the full suite (the owner's PC runs nothing heavy,
owner 2026-11-11). CI is the first full run.
