# F1b float32-frame re-fit + combined-cycle completion — findings

**Version:** silly-kicks 4.128.0 · **Run commit (M):** `b62c1f24a7a9e3361ce416b402ed27da4a59b9e6`
(PR-1 squash-merge on `main`; clean tree on every artifact) · **Date:** 2026-10-05 ·
ADR-106 (float32 frames), ADR-107/108 (native DAS).

All artifacts in this directory are aggregates only — no owner-tier match ids
(`tests/test_research_artifacts_carry_no_ids.py` guards that; the T10 per-match table is deliberately
omitted, spec §8). Numbers below are re-read from the committed `metrics.json`, `receiver_gate.json`
and `hub_smoke.json`, and from the DGX wave logs where noted (scratch evidence; only the numbers enter
this file).

## Why the re-fit exists

ADR-106 stores tracking-frame coordinates as float32 (half the memory) and computes in float64. The
storage rounding exceeds the trained-model feature-contract `atol`, so every frame-geometry bundle is
re-fit on float32-stored frames to make the wheel's weights load against the float32 canonical frames.
The re-fit is not purely "isolate float32": the `default` re-fits also absorb label changes accrued
since their prior training commit (§0.1b). Both are recorded below.

## T10 — feature delta per model (float64 vs float32 frames)

`metrics.json` compares every live feature of every bundled frame-geometry model on the same 179-match
corpus, float64 frames vs float32 frames, at `atol = 1e-6`. `max_abs_delta` is the largest single-cell
delta over all features; `moved` / `unmoved` count features with any cell above `atol`;
`selection_instability.frac` is the fraction of rows whose row-selection (carrier / nearest-player
argmax) flipped between the two storages. Every model has `ok > 0` matches — the T10 refusal guard
(`_refuse_unmeasured_models`, amendment 2 F9/T10) would have aborted the artifact had any model errored
on every match.

| Model | max_abs_delta | moved / total | match status (`ok` / `selection` / other) | selection flip frac |
|---|---|---|---|---|
| ghost_gk | 1.0 | 19 / 26 | 4576 / 176 / 3 `dup_keys` | 0.0 (0 of 1 735 898) |
| ghost_outfield | 1.53e-05 | 12 / 20 | 3580 / 179 | 1.9e-06 (8 only-f64 of 4 170 912) |
| gk_completion | 4.11e-06 | 2 / 8 | 1432 / 179 | 0.0 (0 of 9 566) |
| receiver | 2.0 | 3 / 3 | 537 / 179 | 0.0 (0 of 1 536 852) |
| xcross | 17.74 | 12 / 16 | 2864 / 179 | 4.1e-06 (3 only-f64, 2 only-f32 of 1 209 329) |
| xshot | 5.36 | 26 / 27 | 4833 / 179 | 6.9e-06 (7 / 7 of 2 024 364) |

The large `max_abs_delta` on xcross/xshot/receiver/ghost_gk is a handful of rows near a selection
boundary (a count feature or an argmax tip) crossing when a coordinate rounds; the row-selection flip
fractions (≤ 7e-6, and exactly 0 for ghost_gk/gk_completion/receiver) show the storage rounding almost
never changes which player/frame a feature reads. This is exactly the sub-`atol`-but-above-contract
drift that forces a re-fit rather than a silent reload.

## Re-fit acceptance (pre-registered vs measured)

Pre-registered from the archive's own `3ca609f` extraction (§8); a different count STOPS the run. All
counts matched exactly.

| Model (variant) | corpus | pre-registered | measured | shipped_variant |
|---|---|---|---|---|
| xshot `default` / `position_only` | public (idsse+skillcorner, 17) | 156106 rows / 34649 pos | 156106 / 34649 | public |
| xcross `default` / `position_only` | public (idsse+skillcorner, 17) | 91999 rows / 2849 pos | 91999 / 2849 | public |

**`default` label-change deltas (recorded, never gated against the old labels, §0.1b).** The `default`
re-fits also absorb the label changes since `6e3a132` (ADR-055 GoalMap, ADR-063, the ADR-052 seam).
xshot `default` moved from 34205 → 34649 positives (+444; positive rate 0.2191 → 0.2220). Held-out CV
of the shipped `public` candidate: xshot PR-AUC 0.339, Brier 0.163 (< base-rate Brier 0.173), log-loss
0.497; xcross PR-AUC 0.088, Brier 0.0295 (< base 0.030), log-loss 0.127. All four acceptance gates
(`enough_usable_folds`, `pr_auc_gt_base_rate`, `brier_lt_base_rate_brier`, `log_loss_lt_uniform`) pass.

**ghost `position_only` re-fit vs the prior bundle.** Same 179 games / 1 039 502 samples. CV euclidean
MAE 1.1484 m (float32, M) vs 1.1477 m (float64, `4bda048`) — +0.0007 m. Per provider: Gradient Sports
1.0855 vs 1.0855, SkillCorner 1.2129 vs 1.2098, Sportec 1.7014 vs 1.7000. Acceptance (`overall < 2 m`,
`per-provider < 3 m`, `cross-fold std < 0.5 m`, `size < 15 MB`) all pass. The re-fit is storage-rounding
only; positioning is unchanged.

**gk_completion `skillcorner` (C1 rebundle, 10-match public arm).** `artifact_label = public`,
`all_public = true`, 10 matches / 542 rows. Held-out AUC: overall 0.694, gk_pass 0.740 (floor 0.70),
goalkick 0.461 (n=81). Decision `bundle_skillcorner`. The `default` gk_completion variant is the reused
F1b rebundle (`training_commit 3ca609f`).

## Receiver — widening gate (D7/D8)

The receiver gate (`receiver_gate.json`, `scripts/validate_receiver_widening.py`) decides whether the
327-match widening ships over the committed 30-match corpus.

- **Identification.** `identified = true`, `candidate_reproduces = true`: the 30 identified matches
  reproduce the committed model (re-fit top-1 CV 0.50969 vs the committed criterion 0.50977, within
  `top1_tol` 0.005; `max_rel_param_diff` 0.00116 < `param_rtol` 0.01). Negative control: 20 control
  draws, **0** reproduced the committed model (`n_controls_passing = 0`).
- **Gate (30 → 327).** 297 held-out test matches, 98 954 test passes. top-1 new 0.49577 vs old 0.49691,
  **diff −0.00114**, bootstrap 95 % CI [−0.00214, −0.00011] (seed 0, 2000 resamples).
- **Decision rule (D8):** ship iff point estimate ≥ 0 **and** bootstrap LB95 > −0.01 (δ = 0.01).
  The LB95 (−0.00214) clears the margin, but the point estimate (−0.00114) is below 0, so
  **`ship = false` — the widening is rejected.**
- **Fallback used (spec §7).** The receiver is re-fit at C1 on float32 frames on exactly the 30
  identified matches (`--match-ids-json`) and bundled as the float32 re-fit of the committed 30-match
  corpus (top-1 CV 0.5097), so every frame-geometry bundle is still re-fit per F1b §4.3. The pooled
  Gradient Sports candidate was evaluated and rejected (`keep_pool = false`, margin −0.0221).
  `corpus_visibility = restricted`, `reproducibility = restricted` (the statsbomb corpus is
  manifest-private, D7; ADR-062).

## xcross `default` held-out GS probe (D5c) + record change (D5)

The xcross held-out Gradient Sports probe ran on matches `gradientsports/10502` and
`gradientsports/10503` (Gradient Sports WC2022 is public at source), both held out of every training
fold (`probe_sample_in_training_folds` is `false` for each). Score-differential coverage 1.0,
observed range [−3, 3], `abs_ge_12_count = 0`. `probe_gated_on_held_out = false`. The record now
carries `tf19_ready = false`: the GK |Δ| did not clear `ratio ≥ 2.0 × control` AND the abs-floor
≥ 0.01, so the xcross surface ships but is loudly flagged **not TF-19-ready** (D5 — a recorded record
change, not a silent one).

## D3 — parallel launcher validations

**D3a (DAS map, launcher vs serial).** The production DAS map is launcher-driven
(`_parallel_launch.py --mode das`, nproc 2, per-worker 7.5 GiB cap, 20 GiB headroom): 8 h 01 m wall,
25.2 GB aggregate peak RSS, reducing to `das_native_parity/metrics.json` — **0 rows outside the golden
bounds** across all 980 listed / 895+64+7 scored matches (14 velocity-less SB360 skillcorner entries
excluded and recorded), 0 finite-mask mismatches, 0 direction disagreements over 902 184 compared
frames, D-KEY 0 of 980. An independent serial leg on 31 stratified matches (gradientsports / idsse /
skillcorner) ran in 29 m 16 s at 11.4 GB peak (per-match 44–53 s); the launcher and serial legs reduce
under the same driver, so the shard-split + reduce path changes no DAS value — the golden-bound parity
to the pinned `accessible-space==2.0.15` reference is the verification, not a byte compare of the
(non-deterministic) parquet containers.

**D3b (study fan-out, launcher vs serial).** The paired-study fan-out (15 study tags:
public / sc_extended / full × 5 folds) ran serially (≈ 1 h 10 m) and through the launcher
(cgroup backend, 2 workers, 15 items, 0 relaunched; ≈ 1 h 08 m — the small set gives no wall speed-up,
parity is the point). The two runs are **byte-identical**: `model.json` identical, all 53 feature-cache
/ mmap `X`,`y` shards identical, `SHA256SUMS` identical, and the optuna trial values match trial-for-trial.
Study-worker peak RSS on the small set ≈ 11.9 GB (the 8-match fan-out prep; a small-set figure, not a
full-corpus bound).

## Hub smoke (pre-registered C1 expectations)

`hub_smoke.json` (run at M on the live Hub) matched its pre-registered C1 state exactly:

- `load_refused` = the two sweeper mirrors (`ghost-gk-sweeper-v1`, `ghost-gk-sweeper-position-only-v1`)
  — the Hub still holds the pre-4.111.0 both-axes weights, so they fail closed on the current loader
  (`IntegrityError: chirality mismatch`, the y-mirror signature);
- `cards_mismatched` = the same two sweeper repos;
- `mirrors_mismatched` = all four mirrors (`ghost-gk-sweeper` ×2, `ghost-outfield` ×2): wheel
  `training_commit 3ca609f` vs Hub `adafb72` (sweepers) / `b68328a` (outfields).

All three are cleared by the post-release Hub pushes (Task 22 Steps 11–13). Every frame-geometry repo
that loaded returned finite scores on the float32 canonical frame (`all_finite = true`); the four
HF-only xshot/xcross repos and `ghost-gk-v1` are unchanged.

## `--lock-commit` check

`git diff 6b242cf..M -- silly_kicks/gkdv/_validate.py` (measured at `43ff0dd`, amendment 2): comments
plus the additive `_ARM_DIRECTION_KEY` / `expected_direction_for_arm` (`07a88f6`, PR-S175), which the
sign-off never imports. `ICC_ANCHORS` and `ATT_RELATIVE_ANCHORS` — the only `_validate` names
`run_signoff_power.py` imports — are unchanged.

## Corpus bounds

| Workload | population | token |
|---|---|---|
| DAS parity | 980 listed (gradientsports 64 / idsse 7 / skillcorner 909), 966 scored | owner |
| T10 feature delta | 179 matches | owner |
| xshot / xcross re-fit | the original 17 public (idsse + skillcorner) | public |
| gk_completion skillcorner | 10 public skillcorner | public |
| receiver (identification + fallback bundle) | 30 identified; gate on 327 | owner (statsbomb manifest-private) |
| ghost `position_only` re-fit | 179 games / 1 039 502 samples | owner |

Anchors: the 7 M-refit dirs (xshot ×2, xcross ×2, gk_completion skillcorner, ghost `position_only`,
receiver) carry `run_commit`/`training_commit = M`; the 6 reused dirs (ghost `default` / `sweeper` /
`sweeper_position_only`, ghost_outfield ×2, gk_completion `default`) carry `3ca609f`
(`tests/test_bundled_weights_corpus_policy.py` asserts each dir against its anchor).
