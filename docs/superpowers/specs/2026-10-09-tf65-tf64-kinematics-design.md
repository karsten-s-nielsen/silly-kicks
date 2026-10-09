# TF-65 + TF-64 — kinematics-smoother bake-off, velocity-gap fix, GK detection-gate (design)

**Status:** DRAFT — for independent review (2 reviewers).
**Date:** 2026-10-09.
**Provenance:** `superpowers:brainstorming` (architectural path). Scope + the two pre-registered
decisions (A/B rule, uncertainty) settled with the owner this session. A throwaway ceiling-measurement
spike (scratch, no committed driver, no retrain) sized the prize and is summarised in §3; its arms and
numbers are evidence for the design, not shipped code.
**Cycle shape:** one feature branch `feat/tf65-tf64-kinematics`, no worktrees; brainstorm → spec →
plan → PR. A single batched DGX re-fit covers all three items (owner decision 2026-10-09).
**Version:** a next-free minor, claimed only at commit-prep per the release ritual
(`feedback_no_version_number_until_commit_prep`); re-derive after `git fetch && git merge origin/main`.

---

## 1. Summary

Three co-batched items against one tracking-model retrain, because they rework the same
`tracking/preprocess` velocity path and the same GK/frame models:

1. **TF-65 — kinematics-smoother bake-off.** The foundational velocity/acceleration estimator
   (`smooth_frames` / `derive_velocities`). A pre-registered in-cycle A/B decides among
   **Butterworth**, **Kalman/RTS constant-acceleration**, and a correctness-only **Savitzky-Golay +
   dense-grid floor**. Adds **acceleration** output and **per-frame uncertainty**.
2. **Velocity-gap contamination fix** (lakehouse handoff, verified). The correctness half of the
   smoother path: a **dense-grid reindex** so a non-detection stops contaminating velocity across the
   gap. Lands as the foundation every smoother arm builds on. Plus a **soft `≤ ~40 m/s` plausibility
   guard**.
3. **TF-64 — GK detection-gate wiring** (ADR-109 follow-up). Gate the 14 audited GK outputs + the
   owner-gated `pre_shot_gk_*` family on undetected SkillCorner keepers via `detected_mask`.

**Value-changing** on every velocity-dependent output → one batched retrain (§9) + a Hyrum notice.

### 1.1 In scope
- Dense-grid reindex + segment-edge handling in `smooth_frames` / `derive_velocities` (§4).
- Soft plausibility guard (velocity NaN + count + warning, not a raise) (§4.3).
- Candidate smoothers: Butterworth (ships), Kalman/RTS (new), SG+reindex floor (§5).
- The pre-registered in-cycle A/B that selects the point-estimate smoother (§6).
- Acceleration columns + always-on per-frame uncertainty, decoupled from the point smoother (§7).
- TF-64 GK detection-gate wiring per its approved spec (§8).
- The batched retrain + release + Hyrum notice (§9).

### 1.2 Explicitly OUT of scope (owner-approved 2026-10-09 — not deferrals to launder)
- **Root-cause out-of-pitch / dead-ball ball-position masking** → its own TODO item. It is a
  multi-provider provider-semantics fix (ball tracked out-of-pitch while `ball_state == alive`; measured
  on GS and present on sportec), with thin marginal benefit over the guard, poorly matched to a smoother
  cycle (§4.4). The guard in §4.3 neutralises the downstream symptom this cycle.
- **TF-50** (physical/locomotor metrics) — consumes the acceleration this cycle emits; separate cycle.
- **TF-66 / TF-67** — downstream of TF-64; separate.
- **Butterworth exact-prewarp default flip** — the spike shows it is immaterial at the operating
  cutoff (§5.3); the deferred TODO item is not forced here.

---

## 2. Background

`tracking/preprocess` ships Savitzky-Golay smoothing (`smooth_frames`) and an SG-derivative velocity
(`derive_velocities`), plus a Butterworth path (`_butterworth.py`, TF-58) and a shared
`PreprocessConfig` with per-provider defaults. The velocity path emits `vx`/`vy`/`speed` only — no
acceleration.

**The verified defect (lakehouse handoff, 2026-10-09):** the tracking-frame model is one row per
DETECTED frame; a non-detection is a MISSING ROW, not a NaN-position row. The SG stages operate on ROW
ORDER, ignoring `frame_id` spacing, and `interpolate_frames`'s `max_gap_seconds` cap only sees NaN runs
BETWEEN existing rows — never a missing-row gap. So the last pre-gap and first post-gap detections sit
adjacent in the array; the smoothing/derivative window treats them as neighbours and fabricates a huge
through-gap velocity. Dominant for the ball on broadcast providers (frequent long ball non-detections),
feeding ghost-GK (`ball_vx/vy/speed`), elastic-sync (`ball_speed/accel`) and the per-action linked-frame
ball state. Pitch control is safe (constant `average_ball_speed`).

**Consumers (value-changing, not cosmetic):** the ghost-GK ball-velocity features, elastic-sync, and the
per-action linked-frame ball state. The fix moves those values on broadcast providers → the ghost-GK
re-fit + the batched retrain.

---

## 3. The ceiling-measurement spike (evidence; throwaway)

A scratch harness (not a committed driver; `feedback_input_measurement_is_a_scratch_run_not_a_committed_driver`)
measured six arms on raw frames from the materialized multi-provider corpus (GS broadcast 30 Hz,
SkillCorner broadcast 10 Hz, sportec full-pitch 25 Hz control), token-free. Arms: SG-current (status
quo), SG+dense-grid, Butterworth **linear**, Butterworth **exact-prewarp**, Kalman/RTS, and a GCV
smoothing spline. The two Butterworth arms were identical to 0.1 m/s (§5.3), so the table collapses them
to one `Butterworth` row.

**Sized prize (ball speed, mean across matches, m/s; 6 arms, Butterworth linear/prewarp collapsed):**

| arm | GS p99 | GS max | GS frac>40 | SK max | sportec max |
|---|---|---|---|---|---|
| SG-current (bug) | 46 | 626 | 1.17 % | 188 | 370 |
| SG + dense-grid (floor) | 32 | 1674\* | 0.63 % | 52 | 408 |
| **Butterworth** | 22 | **185** | **0.10 %** | 36 | 56 |
| **Kalman/RTS** | 25 | 298 | 0.25 % | 44 | 92 |
| GCV spline | 31 | 641 | 0.57 % | 47 | 262 |

**Decisive trace (GS, p1, the handoff case — a ~930-frame ball gap; raw position barely moves):**
SG-current fabricates 88 → 626 m/s into the gap; Butterworth holds 3–5 m/s (correct); Kalman bounds it
(~50, with growing uncertainty); SG+dense-grid still spikes to ~380 at the pre-gap segment edge.

**Three separable effects the spike established:**
1. **Dense-grid reindex = mandatory correctness.** The change concentrates AT gaps (velocity RMS Δ vs
   current: GS 62–71 m/s near-gap vs ~0.4–1.4 elsewhere; sportec control ≈ no-op, ~0.025 RMS) — it
   targets the bug, not a global perturbation. Lands regardless of the smoother.
2. **A filter swap is materially cleaner than SG+reindex alone.** Butterworth cuts GS frac>40 ~12×;
   Kalman ~5×. SG+reindex leaves intermittent pre-gap **segment-edge spikes** (the 1674\* max) — an
   edge-handling gap, see §4.2.
3. **Acceleration capability.** All arms can emit a 2nd derivative, but only Butterworth/Kalman give
   plausible acceleration; SG-deriv2 and spline are noise-dominated (GS frac |accel|>10: SG/spline
   0.48–0.50 vs Butterworth 0.22, Kalman 0.37). Kalman uniquely emits per-frame **uncertainty**.

\* The SG+dense-grid max is inflated by a harness edge-handling choice (valid frames within the
filter radius of a segment boundary were not NaN'd); §4.2 fixes this in production. Judge SG+reindex on
p99/frac>40 (both improved), not max.

**Conclusions carried into the design:** the gap fix is mandatory; the smoother choice matters beyond
the gaps and warrants the in-cycle A/B; GCV spline is dominated (dropped — modest tail, noisy accel);
Butterworth and Kalman are the finalists; SG+reindex is the correctness-only floor; exact-prewarp ≈
linear at the operating cutoff.

---

## 4. Correctness foundation (smoother-agnostic; commit 1)

Lands first because every arm — and the floor — needs it, and the spike must have been (and the A/B
will be) measured on gap-corrected velocities to be fair. Pure (pandas in, pandas out), additive, no
contract removed.

### 4.1 Dense-grid reindex

For each `(game_id, period_id, is_ball, player_id)` group:

1. Reindex to the group's contiguous `frame_id` range; missing frames become NaN-position rows. Output
   row-count is unchanged (reindex is internal; only the group's own frames are returned). Duplicate
   `frame_id` within a group (GS emits them, `reference_gradientsports_duplicate_frames`) are
   de-duplicated **keep-first** before reindex — matching the preprocess precedent (`_elastic_sync.py:118`,
   `features.py:6039`); DAS's `raise`-on-duplicate (`_das_pack.py:140`) guards a stricter invariant and
   does not apply here (owner-approved 2026-10-09, recorded in the cycle ADR).
2. **Bridge** interior NaN runs ≤ `max_gap_seconds` (linear), leave runs > `max_gap` NaN — reusing the
   existing `max_gap_seconds` policy, no new policy.
3. **Segment** the series at every NaN run > `max_gap` so no smoothing/derivative window spans a big
   gap.
4. Run the active smoother per segment; re-NaN the originally-missing rows.

A non-detection now behaves exactly like a NaN-position run the `max_gap` cap already handles → velocity
is NaN across a > `max_gap` gap, and `x_smoothed` is fixed too (not only velocity). Idempotent;
byte-identical on contiguous single-detection-run groups (the sportec control confirmed ≈ no-op).

### 4.1a Group-key unification (`game_id` on all three) — owner-approved 2026-10-09

`smooth_frames` keys on `[game_id, period_id, is_ball, player_id]` (A-31 fix: a two-game frame must not
smooth an entity's series across the game boundary). Its siblings **do not**: `derive_velocities`
(`_velocity.py:16`) and `interpolate_frames` (`_interpolation.py:15`) key on `[period_id, is_ball,
player_id]` — the identical latent cross-game bridge, live in every adapter's velocity path. Add
`game_id` to **both** (mirror the A-31 rationale comment); `interpolate_frames` stays (public surface /
PRIVATE_CONSUMERS — fix the key, do not remove it). Single-game frames are byte-identical (`game_id`
constant → identical groups); only multi-game frames are corrected — which is the bug fix.
- **Tests:** (a) single-game byte-identity pre/post the key change on each of the two; (b) a two-game
  frame proves no cross-game bridge at the seam.

### 4.1b Two gap-fill stages — complementary, not overlapping

The dense-grid reindex (§4.1) and the standalone `interpolate_frames` target **different gap forms**:
`interpolate_frames` fills **NaN-position runs between existing rows**; the reindex handles **missing
rows** (the actual non-detection bug `interpolate_frames` never sees). Both honour the same
`max_gap_seconds` cap. Default-pipeline order: `interpolate_frames` → `smooth_frames`/`derive_velocities`;
the reindex is internal + idempotent w.r.t. already-filled frames, so no double-fill. Document this
ordering + the shared cap in the module docstrings.

### 4.2 Segment-edge handling (the spike's 1674 finding)

At a segment boundary, an SG derivative uses the asymmetric edge polynomial and can spike on short
pre-gap segments. The production rule: **NaN the derived velocity (and acceleration) within the filter
radius of each segment boundary.** This trades a little coverage near every gap for honesty (no edge
spike). Padded filters (Butterworth `sosfiltfilt`) and the Kalman covariance are far less edge-sensitive
(spike evidence), but the edge-NaN rule applies uniformly so the floor is honest too.

### 4.3 Soft plausibility guard

After velocity derivation, a `max_plausible_speed` guard (default ~40 m/s): **NaN** the offending
velocity, **count** it, and emit a dedicated `PlausibilityWarning` (its own category, never an umbrella;
`stacklevel=2`). **NOT a hard raise** — out-of-pitch ball frames legitimately exceed 40 m/s (~6 % of GS
ball rows, §4.4), so a raise would fire every match. The guard is the in-cycle safety net that
neutralises the downstream symptom from BOTH the gap bug (fixed at source by §4.1) and the out-of-pitch
source (§4.4, out of scope) — no model consumes an implausible ball velocity after it.

An analogous acceleration plausibility bound applies to the new `accel` column (§7).

### 4.4 What the guard does NOT fix (→ separate TODO)

The spike surfaced a distinct implausibility source: the ball is tracked **out-of-pitch** (measured
x ∈ [-14, 122], y ∈ [-7, 80]; SPADL pitch 105×68) while `ball_state == alive` for 100 % of ball rows —
no existing column flags it, and it is present on sportec too (not GS-only). That is genuine out-of-play
tracking, not a gap, so the reindex cannot fix it; a smoother only bounds it. The guard NaNs the
resulting velocity (symptom handled). The **root-cause position masking** (dead-ball / out-of-pitch
detection with no provider flag, multi-provider, and an interaction to verify with
`add_restart_coordinates` / ADR-025 restart geometry) is a provider-semantics fix with thin marginal
benefit over the guard → **its own TODO item**, not this cycle (owner-approved).

### 4.5 Repro tests
- A unit group with two valid-position runs separated by a `frame_id` gap > `max_gap` (no NaN
  positions): assert the derived velocity at the boundary is NaN (not a spike).
- The named real repros: the GS p1 ball gap at frames 57169–58109 and the WC2022 case named in the
  `_velocity.py` comment.
- A physical-plausibility assertion on real fixtures (built `speed ≤ ~40 m/s` after the guard).

---

## 5. Candidate smoothers

The smoother is selected by `PreprocessConfig.smoothing_method` (already the seam). All arms consume the
§4 dense-grid + edge handling; only the per-segment estimator differs.

### 5.1 Butterworth (ships; `_butterworth.py`)
Zero-phase dual-pass (`sosfiltfilt`), Winter cutoff correction. Velocity/acceleration via finite
differences of the smoothed position. Cleanest tail in the spike; cheapest; no native uncertainty
(supplied by §7).

### 5.2 Kalman/RTS constant-acceleration (new)
A constant-acceleration state-space `[p, v, a]` per coordinate, forward Kalman + RTS backward smoother.
Native gap handling: on a missing frame the update is skipped (predict-only) so covariance grows through
the gap — the honest-gap correctness win. Emits position/velocity/**acceleration** and the state
**covariance** (the uncertainty source, §7). Process/measurement noise are frozen per-provider params
(`for_provider` promotion). ~1.7 s/group in the spike — not a bottleneck.

### 5.3 Savitzky-Golay + dense-grid (the floor)
The incumbent SG smoother on the §4 dense grid with §4.2 edge-NaN. Correctness-only: it fixes the gap
bug without a filter swap. The pre-registered fallback if no finalist beats it (§6).

### 5.4 Dropped / not-default
- **GCV smoothing spline** — dominated in the spike (modest tail, noise-dominated acceleration); dropped.
- **Butterworth exact-prewarp** — ≈ linear at the operating cutoff (spike: identical to 0.1 m/s); the
  default stays linear-Winter; the prewarp flag remains available. The deferred flip-default TODO is not
  forced here.

---

## 6. The in-cycle A/B gate (pre-registered)

Selects the **point-estimate** smoother. **Pre-registered in this spec and the cycle ADR, committed
before the A/B runs** — the rule, endpoints, thresholds, corpus, seeds, and model subset are fixed in
advance; no post-hoc metric selection. It is a scratch run (not a committed driver), affordable only
because a retrain is being eaten anyway. The winner is not known at spec time by design — this is an
explicit pre-registered gate, not an open TBD; the fallback is named.

### 6.1 Protocol
- Arms: Butterworth, Kalman/RTS, and the SG+dense-grid floor.
- Identical model hyperparameters and seeds across arms; the ONLY varying input is the smoother →
  clean causal attribution.
- **Split by MATCH** (train/eval), never by frame — no leakage.
- Cost control: the A/B trains a pre-registered **representative velocity-sensitive subset** (ghost-GK,
  plus xshot/xcross as guardrail models), not all 6 F1b models; the full 6-model retrain runs once under
  the winner (§9).
- **Corpus:** the MAE primary runs on the **full 179-match corpus** (SK 108 / GS 64 / idsse 7). The
  density-NLL confirmatory runs on a **fixed, pre-registered 10-match subset** = per provider, the
  lexicographically-first shard basenames: **5 skillcorner + 4 gradientsports + 1 idsse** (deterministic,
  spans all three providers; no post-hoc subset selection). The subset is chosen by rule, not enumerated,
  so the pre-registration carries no provider match-id list.
- **Materialize cost:** arm materialization runs the full C1 `smooth_frames`+`derive_velocities`
  (`derive_velocities` ≈ 99 s/match; it has no uncertainty-skip flag, and velocity needs full-resolution
  neighbor frames, so the cost is irreducible). Full-179 × 3 arms ≈ ~21h, materialize-dominated.
- Report the full arm × endpoint matrix (measure both sides), not only the winner.

### 6.2 Primary endpoint (RE-AMENDED 2026-10-09, owner-approved, pre-run — HYBRID)
The most direct velocity consumer is **ghost-GK** (consumes `ball_vx/vy/speed`). Scored two ways,
match-level CV, identical hyperparameters/seeds across arms:

- **PRIMARY — decision (full corpus, 179 matches: SK 108 / GS 64 / idsse 7):** ghost-GK out-of-sample
  **Euclidean MAE** (metres) from `GhostGkModel.predict_mean`. This is the decision driver — it carries
  full-corpus match-level power (`predict_mean` ≈ 0.2 ms/row, tractable at corpus scale).
- **CONFIRMATORY — density guard (fixed 10-match subset, §6.1):** ghost-GK **density-NLL**
  `−log p(true keeper cell | predicted density)` from `GhostGkModel.predict_density`, **KDE backend
  pinned `kde_backend="vectorized"`** (THE exact numpy closed-form raw grid — platform-independent,
  numba-free, the backend the KDE golden compares against; `predict_density` exposes no `"exact"`
  literal, so the exact raw grid IS `"vectorized"`). A strictly-proper scoring rule over the full
  predictive distribution (rewards accuracy AND calibration AND sharpness — the shape GKDV counterfactual
  and rest-defense ghost frames consume, which a point metric ignores). The MAE winner must NOT regress
  density-NLL vs the floor on the subset (§6.3). Each test match's mean density-NLL is estimated from a
  **seeded 150-row eval subsample** (`predict_density` is ~2 s/row even at the subset's ~88k-row train
  fold; a full-eval confirmatory is infeasible — the subsample bounds it to ~hours).

**Why hybrid (measured in-cycle, not reasoned):** density-NLL via `predict_density` is a leaf-match KDE,
**O(n_train) per eval row** (`_leaf_match_weights` is an `(n_query, n_train, n_trees)` broadcast). Measured
148 ms/row (cpu-numba) / 277 ms/row (vectorized) at n_train = 11.7k (one match) → **~23 s/row** at a
full-179 training set (≈ 1.8M labeled rows); full-corpus density-NLL is infeasible (cost scales ≈ L²·K² in
rows/match × matches — tractable only at K ≲ 12 matches, which lacks match-level power). MAE carries the
decision at full-corpus power; density-NLL confirms calibration on a bounded subset. [Supersedes the
earlier 2026-10-09 density-NLL-**primary** amendment, which was computationally infeasible at corpus
scale. Original §6 "log-loss" was also wrong — ghost-GK is a position/density model, not a classifier.]

### 6.3 Superiority test (the noise-exceeding margin)
- **PRIMARY (MAE, full corpus):** a finalist beats the floor iff the **one-sided 95 % paired match-level
  bootstrap** CI of `[floor − finalist]` per-match mean ghost-GK **MAE excludes 0** (lower MAE is better;
  B = 2000 resamples, fixed seed). A statistical margin, not a hand-picked constant.
- **CONFIRMATORY (density-NLL, fixed subset):** the chosen arm's per-match mean density-NLL on the
  10-match subset must **not be worse than the floor's** (no CI required at K = 10; a density-NLL
  regression DISQUALIFIES the arm and triggers re-evaluation — guards against a point winner that wrecks
  calibration).

### 6.4 Guardrails (any regression vs the floor ⇒ the arm is disqualified)
- **Implausible-kinematics rate** (>12 m/s, >40 m/s, |accel| > 10 m/s²) not worse — FULL corpus (free,
  from materialize). Directly measures the fabricated-velocity failure mode the fix targets.
- **Velocity coverage** (non-NaN fraction) not lower than the floor by > 0.5 pp — FULL corpus; stops an
  arm "winning" by dropping hard frames.
- **Acceleration plausibility** (p99 within a physical bound) — FULL corpus.
- **xshot / xcross** log-loss not worse — on a **fixed 30-match subset** = per provider, the
  lexicographically-first shard basenames (17 skillcorner + 10 gradientsports + 3 idsse), stratified by
  the corpus mix. A non-regression guardrail needs representativeness, not the whole corpus; positives are
  ample (xshot ≈ 3130/match, xcross ≈ 212/match). Full-179 would add ~18.6h (measured +124 s/match ×
  179 × 3); the subset adds ~3.1h. Models use default params (no HPO) — identical across arms, so the
  arm-to-arm log-loss delta is the signal.
- **DAS → near-term-danger/shot AUC — DEFERRED (owner-approved, this cycle).** Full-179 × 3 arms is
  multi-day (DAS compute per action-frame), beyond the cycle's compute envelope. The frame-level
  implausibility/coverage guardrails above already gate the DAS-facing player-velocity consumer at the
  level the fix changes; a dedicated DAS-AUC sweep is a bounded follow-up if a finalist is marginal.

### 6.5 Decision + fallback
Pick the arm with the largest statistically-distinguishable **MAE** improvement (§6.3 primary) that
breaches no guardrail (§6.4) AND does not regress the density-NLL confirmatory (§6.3). Overlapping MAE
CIs ⇒ tiebreak by DAS validity, then capability (Kalman's coherent uncertainty, §7), then simplicity.
**If no finalist clears the §6.3 MAE test without a guardrail/density breach ⇒ ship the floor**
(SG+dense-grid; correctness-only, no filter swap).

---

## 7. Acceleration + per-frame uncertainty

Acceleration is in scope (gates TF-50). Uncertainty is **decoupled from the point-estimate smoother and
shipped unconditionally**, so the schema is fixed regardless of the A/B outcome.

- **Point columns:** `accel_x`, `accel_y`, `accel` from the A/B-winning smoother (component columns are
  `accel_x`/`accel_y`, **NOT** `ax`/`ay` — impl found `ax`/`ay` already in use as `_kernels.py` triangle
  anchor-coordinate columns in `add_action_context`'s merge; a frame `ax`/`ay` collides there. Rename
  owner-approved 2026-10-09). Float32 storage / float64 compute (ADR-106); NaN-preserving.
  **ReflectionKind (ADR-045):** `accel_x`/`accel_y` are vector components (sign-flip under reflection,
  like `vx`/`vy`, enumerated in `_reproject_rows`); `accel` is a magnitude (invariant, like `speed`);
  the variance columns are invariant. Mirror-invariance + context-reprojection gates cover them.
- **Uncertainty columns:** emitted from an **always-run constant-acceleration Kalman/RTS pass** — the
  per-frame state-variance diagonal `pos_var` / `vel_var` / `accel_var` (float32 storage). The covariance
  honestly encodes occlusion-distance (grows through gaps) + local noise. Cheap (~1.7 s/group). A scalar
  `kinematic_confidence` derived from these is an optional plan-time convenience, not the contract.
  - If **Kalman** wins the point A/B ⇒ point + uncertainty are one coherent estimator (ideal; also the
    §6.5 tiebreak).
  - If **Butterworth / floor** wins ⇒ point = winner, uncertainty = the parallel RTS covariance,
    **documented** as a confidence proxy (not the point estimator's own variance).
- **Contracts touched:** `reflection.py` (`ReflectionKind`: `accel_x`/`accel_y` vector,
  `accel`/`pos_var`/`vel_var`/`accel_var` magnitude-invariant) + `utils._reproject_rows`; `NOTICE`
  (Savitzky-Golay 1964, Winter 2009, Kalman/RTS citations); and the dtype-invariance / mirror-invariance
  / liveness / float32-storage gates. **NOT** `TRACKING_FRAMES_COLUMNS` / `metric_contracts` /
  `feature_glossary` — derived kinematic columns follow the `vx`/`vy` convention (preprocess-added, not
  declared; registration struck, owner-approved 2026-10-09; see §11). Purity gated by
  `tests/tracking/preprocess/test_preprocess_purity.py` (PURITY_ENTRIES is `add_*`-exact).
- `derive_velocities` keeps its loud raise for missing smoothed inputs (principle of least surprise);
  the new columns are documented additions (output schema change → Hyrum notice, §9).

---

## 8. TF-64 — GK detection-gate wiring (commit 3)

Per the approved spec `docs/superpowers/specs/2026-10-01-gk-detection-gate-design.md` (approved over 3
`/review-spec` rounds; re-anchor its file:line pins at plan time). Wire the shipped ADR-109
`detected_mask` primitive into the **14 audited keeper-position-dependent outputs** (8 restdefense GK
columns, `gk_decision` Tier B, gkdv, 4 `gk_influence`) **plus the owner-gated `pre_shot_gk_*` family**
(the approved spec's test-1 writes "the 14 + the gated `pre_shot_gk_*` family (not ~13)"), so they
honest-NaN / count-drop on an undetected SkillCorner keeper (detection ≈ 17.6 % of live frames) instead
of consuming ~80 %-extrapolated positions. The **4 sweep-floor candidates** (`_gk_geometry.py`,
`shot_stopping/_compute.py`, `positioning/_compute.py`, `_cover_shadows.py`) are dispositioned at plan.
Fail-closed default + the general `assume_observed` opt-out; glossary caveats. **Mechanical test-8
DERIVES the keeper-position-reader population from code (ADR-056)** and fails on any member neither
detection-gated nor in a reasoned out-of-scope allowlist (`gk_decision` Tier A native = caveat-only,
owner-signed-off). TF-58 is **landed** on main (#270, `75d9003`) — rebase `_provider_visibility.py`
onto it; no ordering race. Value-changing on SkillCorner GK outputs → part of the Hyrum notice + the
cross-repo lakehouse re-materialise (§9).

---

## 9. Retrain + release

**After** the A/B names the winner, one batched retrain. The in-repo vs cross-repo split is explicit:

**In-repo (this cycle's deliverable, committed in C2):**
- The 6 F1b frame-geometry models (the velocity/accel-dependent tracking models) — retrained bundled
  weights + `SHA256SUMS`.
- **Corpus** DAS re-fit under the new velocities (the lakehouse/corpus DAS values move). NOTE: the
  committed **unit** DAS parity oracle (`tests/tracking/_das_golden.py` / `scenes_frames.csv`) carries
  `vx`/`vy` as fixture inputs and `_das_pack.py:84` reads them directly — it never runs the preprocess
  path, so it is **byte-identical** under this cycle and is NOT regenerated. C1's golden/snapshot
  blast-radius audit confirms this (plan).
- The T10 baseline.

**Cross-repo (handed to the owner; NOT an in-repo deliverable):**
- Lakehouse re-materialise (frames + the GK-output changes from TF-64) — consumer-side, runs where the
  lakehouse lives; the `<5` version pin holds.

**Release:** a next-free minor, number claimed at commit-prep (release ritual; non-squash admin merge;
tag only after post-merge CI green; owner publishes). **Hyrum notice:** velocity values move on
broadcast providers, new `accel_x`/`accel_y`/`accel` + uncertainty columns, and the TF-64 GK-output honest-NaN
behaviour — all observable output changes. HF model-card pushes are the session's post-release job.

---

## 10. Phasing & commit discipline

One feature branch `feat/tf65-tf64-kinematics`, off the default branch, no worktrees. Each commit is a
**coherent, fully-tested state**; no micro-commits. **No commit, push, tag, or retrain without the
owner's explicit per-commit approval** — the gate is explicit below.

| # | Content | Gate |
|---|---|---|
| 1 | §4 correctness foundation (dense-grid + edge-NaN + soft guard + §4.1a `game_id` key unification) + §7 accel/uncertainty plumbing, smoother-agnostic; full suite green | **owner per-commit approval** |
| — | DGX A/B run (§6) — scratch, not a commit | owner go to launch |
| 2 | A/B winner as default smoother + its productionised arm + accel(+coherent uncertainty if Kalman) + retrained bundled weights + `SHA256SUMS` | owner approval |
| 3 | §8 TF-64 GK detection-gate wiring | owner approval |
| N | version bump + `CHANGELOG` (`PR-Snnn`) + Hyrum notice (commit-prep) | owner approval |

The A/B result and the §6.5 decision are recorded in the cycle ADR between commits 1 and 2.

---

## 11. Testing & CI

CI-faithful (`feedback_run_the_ci_faithful_pytest_invocation`): full `pytest -m "not e2e"` +
`ruff check silly_kicks/ tests/ scripts/` + `ruff format --check` (independent gates) + bare `pyright`;
then `/final-review`.

New-surface gates the additions must satisfy (several fail only in the full suite —
`feedback_new_sibling_metric_package_ci_gates`):
- Derived kinematic columns (`accel_x`/`accel_y`/`accel` + `pos_var`/`vel_var`/`accel_var`) follow the
  `vx`/`vy` convention: **NOT declared** in `TRACKING_FRAMES_COLUMNS` / `metric_contracts` /
  `feature_glossary` (preprocess-added, `schema.py` precedent) — declared only in `reflection.py`. The
  409-test contract gate confirms this is correct (registration sub-tasks struck, owner-approved
  2026-10-09). Preprocess purity is CI-gated by a dedicated `tests/tracking/preprocess/test_preprocess_purity.py`
  (`smooth_frames`/`derive_velocities`/`interpolate_frames` stay PURE) — NOT `PURITY_ENTRIES`, which is
  `add_*`-exact (ADR-056 meta-gate rejects non-`add_*`).
- `*_xfns` leak guard: acceleration is a kinematic (no post-contact `result_id` read) — safe to expose;
  confirm no arm reads its own outcome.
- Reflection (`accel_x`/`accel_y` vector, `accel`/`*_var` magnitude) + **mirror-invariance** from both
  sides; context `_reproject_rows` enumerates `accel_x`/`accel_y`; float32-storage / float64-compute gate;
  dtype-invariance.
- **Uncertainty honest-gap property (not just liveness):** a test asserting `pos_var`/`accel_var`
  **strictly rises** with distance-to-nearest-detection — measured at a gap-adjacent frame vs a
  dense-interior frame on the same fixture. Non-constant liveness is too weak; this asserts the property
  that justifies the columns (the TF-50 consumer).
- **Consumer NaN-tolerance:** a named test that ghost-GK (`ball_vx/vy/speed`) + elastic-sync
  (`ball_speed/accel`) emit NaN-not-crash on the ~6 % of ball frames the §4.3 guard NaNs.
- **§4.1a key-unification:** single-game byte-identity (pre/post) on `derive_velocities` +
  `interpolate_frames`; two-game no-cross-game-bridge at the seam.
- SB360 freeze-frame verdict for any new `add_*` surface; the registry anti-rot meta-assertion.
- The §4.5 repro + physical-plausibility tests; a discriminating-power precondition for every new
  liveness fixture (`feedback_invariance_test_needs_discriminating_power`).
- NOTICE entries for Savitzky-Golay, Winter, and Kalman/RTS (academic-attribution discipline).
- Impl doc fix: `_smoothing.py:106` `method` docstring lists only `{"savgol","ema"}` — add
  `butterworth` (already a supported method).
- No silent skips on required testing; mirror ci.yml's pytest invocation exactly (never a package-scoped
  subset).

---

## 12. Risks & open questions

- **The A/B outcome** is the one decision deferred by design — pre-registered in §6 with a named
  fallback (the floor), so it is a gate, not an open requirement.
- **Kalman provider noise params** (`for_provider`) need tuning; frozen as params, validated against the
  §6.4 acceleration-plausibility guardrail.
- **BW-point + RTS-covariance coherence** (§7) — documented as a confidence proxy; acceptable, flagged.
- **TF-64 rebase** of `_provider_visibility.py` onto landed TF-58 — mechanical, covered by its spec.

---

## 13. References
- The lakehouse velocity-gap contamination handoff (2026-10-09) + the GS diagnostic/root-cause memos.
- TF-64 approved spec: `docs/superpowers/specs/2026-10-01-gk-detection-gate-design.md`;
  ADR-109 primitive: `docs/superpowers/specs/2026-09-27-detection-primitive-design.md`.
- ADR-025 (restart geometry), ADR-045 (reflection), ADR-056 (derived population gate),
  ADR-098 (metric_contracts), ADR-103/106/107/108 (frame dtypes, float32 storage, native DAS),
  ADR-033 (purity). TODO rows TF-65 / TF-64 / TF-50.
- Savitzky & Golay (1964); Winter, *Biomechanics and Motor Control of Human Movement* (2009);
  Kalman (1960) + Rauch–Tung–Striebel (1965).
