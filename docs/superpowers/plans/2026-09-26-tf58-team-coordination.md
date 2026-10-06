# TF-58 Team-Coordination Dynamics — Implementation Plan

> **Execution.** Inline, in one session, on the owner's chosen implementation model (the owner's routing rule:
> Opus 5.5 wrote this plan and the spec; it does not implement). Do **not** dispatch subagents without the owner's
> explicit approval. Steps use checkbox (`- [ ]`) syntax. Every task ends green and lint-clean. **No task commits.**
> There are exactly two commits, each behind an explicit owner-approval stop (§"Commit 1", §"Commit 2").

**Goal:** Ship `silly_kicks.coordination` — Hilbert relative phase, lagged cross-correlation, vector coding, spectral
median frequency, Welch coherence, cluster phase with sample entropy, and the relative stretch index, each with a
surrogate chance baseline — plus the seam changes, the three corpus drivers (D1 derive, D2 calibrate, D3 validate) and
the in-cycle validation artifact, exactly as the approved spec defines them.

**Architecture:** Hexagonal. `coordination/_kernels/` holds pure numpy/scipy kernels (optional `numba`, parity-gated).
`coordination/_signals.py` is the one numeric port from long-form frames to dense, filtered, resampled,
segment-indexed arrays. `coordination/_compute.py` turns signals into per-window tables. Two seams outside the package
change: `tracking/preprocess` gains a zero-phase Butterworth filter, resampling and residual analysis, and a new
vectorised collective-variable kernel (`tracking/_collective.py`) becomes the single definition behind
`compute_team_shape` and `compute_defensive_line`.

**Tech stack:** Python (CI: 3.10 → pandas 2; ≥ 3.11 → pandas 3; local `.venv` is py3.14 / pandas 3), numpy ≥ 2, scipy
(`scipy.signal`, `scipy.spatial`, `scipy.stats`), pandas, optional `numba`, ruthless-efficiency ≥ 0.7.0 (drivers and
calibration only), pytest, ruff, pyright.

**Spec (the bar):** `docs/superpowers/specs/2026-09-26-tf58-team-coordination-design.md` — owner-approved 2026-09-26;
independent reviews: r1 APPROVE WITH FOLLOW-UPS, r2 APPROVE, r3 APPROVE
(`D:\Development\_reviews\2026-09-26-tf58-team-coordination-spec*.md`). `§x.y` below always means a spec section.
Read the whole spec before Task 0. Where this plan and the spec disagree, the spec wins: stop and ask the owner.

**Prerequisite (not built here):** ruthless-efficiency 0.7.0 with `GridSearchStrategy` and required
`StoreConfig.objective_id`, built by a separate session (handoff `D:\Development\_handoffs\ruthless-grid-strategy-handoff.md`;
its implementation was approved at review round 3, `D:\Development\_reviews\2026-09-26-grid-search-strategy-impl-r3.md`).
**Released:** `pip index versions ruthless-efficiency` lists `0.7.0` (checked 2026-09-26). Tasks 1–20 do not need it.
Tasks 21–22 install it from PyPI (Task 21 precondition). Commit 1 is proposed only after 0.7.0 is on PyPI (§13 item 1),
which now holds.

---

## Owner rulings (RULED 2026-09-26)

Plan-time investigation found five points where the approved spec is silent, internally unreachable, or needs a value
it does not state. Each carried a gold-standard recommendation. The independent plan review (r1) concurred with all
five.

**The owner approved all five on 2026-09-26, together with three strengthening amendments:**
- **A1** (to R1): a `computed_nonconverged` surrogate token.
- **A2** (to R3): an exact collinearity predicate in the hull kernel.
- **A3** (to R4): pooled, corpus-derived base defaults, so that no interim placeholder survives into a release.

The plan below is written to the rulings and the amendments. The "If declined" notes are kept only as a record of the
alternatives that were considered.

### R1 — Surrogate columns can be NaN with no recorded reason

**Finding.** §5 goal 3 requires "every NaN carries a `*_source` reason from a closed vocabulary". The per-method
`*_source` column describes the observed metric. The surrogate triple (`*_surrogate_mean`, `*_percentile`, `*_excess`)
can be NaN while the observed metric is `scored`:
- `n_surrogates = 0` (every D2 pass, §7.9);
- the time-shift is impossible: a continuous segment (or a player run, for cluster phase) shorter than `2τ + 1` samples
  cannot be shifted by at least `τ = min_shift_s × fs` (§7.9);
- the observed metric itself is not `scored`.

**Recommendation.** Add one provenance column per surrogated method group: `coord_rp_surrogate_source`,
`coord_xc_surrogate_source`, `coord_vc_surrogate_source`, `coord_coh_surrogate_source` (on `COORD_PAIR`),
`coord_cluster_surrogate_source` (on `COORD_CLUSTER_TEAM`) and `coord_team_sync_surrogate_source` (on
`COORD_TEAM_SYNC`). They take values from a new closed vocabulary, `COORD_SURROGATE_SOURCE_VALUES = ("computed",
"computed_nonconverged", "disabled", "segment_too_short", "not_scored")`. They are provenance columns: in `*_COLUMNS`,
not in `*_METRIC_COLUMNS`, not in the glossary (the `coord_detection_source` precedent, §7.12). The rule for each
value is in Task 17's "Surrogates" bullet.

**Ruling: APPROVED with amendment A1.** `computed_nonconverged` marks a row whose surrogate set includes at least one
IAAFT draw that hit `iaaft_max_iter` before converging. The report counter alone would leave such rows
indistinguishable from fully converged ones. The value is reachable only with `surrogate_method="iaaft"` (opt-in).

**If declined (not taken):** delete the surrogate-source rule in Task 17 ("Surrogates" bullet), every `*_surrogate_source`
entry and `COORD_SURROGATE_SOURCE_VALUES` in Task 14, and `test_every_surrogate_token_reachable`. §5 goal 3 then
needs an explicit owner waiver for surrogate NaNs, recorded in the ADR.

### R2 — Spectral and coherence liveness is unreachable on the committed real fixtures

**Finding (measured on this tree).** The longest committed real tracking fixture is
`tests/datasets/elastic_sync/j03wmx_slice/frames.parquet`: 248 s of one Sportec period. The sportec and gradientsports
`realistic.parquet` fixtures are 20 s; the SkillCorner and Metrica `lakehouse_derived.parquet` fixtures are raw bronze,
not frames. The spectral median frequency needs a slice of at least 2 periods of `band_low` (545 s at the interim
band, §7.8.4). Coherence needs at least 4 Welch segments at 50% overlap (1,000 s at the interim 400 s segment, §7.8.5).
§9.5 asks for every metric column to be live on the committed provider fixtures, which cannot hold for
`coord_median_freq_cpm` or the four `coord_coh_*` metric columns.

**Recommendation.** Commit one real, public, long-enough fixture. Use an IDSSE (DFL open data, Bassek et al. 2025,
CC BY 4.0) first half, cut to its first 1,200 s, decimated by 2 (12.5 Hz), with only the `TRACKING_FRAMES_COLUMNS` the
package reads. Add a generator script that reproduces the parquet byte-for-byte (the ADR-056 fixture rule) and a
`NOTICE`/fixture README attribution. The size target is ≤ 6 MB; the generator records the actual size. The liveness
test then covers every metric column on real data. A synthetic long match still backs the kernel ground-truth tests.

**Ruling: APPROVED.** The ≤ 6 MB size is an estimate, not a measurement. If the generator measures more, it stops
for the owner (Task 18 Step 5).

**If declined (not taken):** Task 18 Step 7 runs spectral and coherence liveness on the synthetic long match only. A precondition
test proves the committed real fixtures are shorter than both minima. Real-data liveness for those five columns moves
to the D3 artifact (Task 23 asserts every column live on the corpus). Delete Task 18 Steps 5–6.

### R3 — The hull-area bound is undefined when Qhull fails on precision-flat input

**Finding.** `compute_team_shape` maps `QhullError` to `0.0`. Qhull raises for exactly collinear points **and** for
nearly collinear points that are flat within its own precision tolerance. The spec pins the new kernel to "exactly 0.0
when every orientation determinant is exactly zero" and "≤ 1e-9 relative" elsewhere (D12, §7.5, §12). On a
precision-flat, not exactly collinear, frame the reference is `0.0` and the new area is a rounding-level positive
number, so the relative bound is undefined (division by zero).

**Recommendation.** Keep the spec's algorithm unchanged. Gate the precision-flat case with an absolute bound of
`≤ 1e-9 m²` (at least nine orders of magnitude below any real team hull). Everywhere Qhull succeeds, the relative 1e-9
bound applies unchanged. Record this bound in the ADR and CHANGELOG sentence on the hull change. Real tracking data
never produces a precision-flat outfield hull; the adversarial test planted in Task 2 is the only producer.

The bound is principled, not arbitrary. Shoelace rounding at pitch-scale coordinates (at most about 110 m, n ≤ 14)
is about `n · 2⁻⁵³ · (110 m)²` ≈ 1e-11 m², so 1e-9 m² leaves about 100× margin.

**Ruling: APPROVED with amendment A2.** Testing `det == 0.0` in floating point is not an exact collinearity test:
- rounding can make exactly collinear inputs look non-collinear, and the kernel would return a tiny positive area
  where the geometric truth is 0.0;
- rounding can also make non-collinear inputs look collinear.

The hull kernel therefore decides collinearity with an **exact orientation predicate**, the computational-geometry
standard (Shewchuk 1997, adaptive-precision orientation):
- a vectorised floating-point filter with Shewchuk's static error bound `ccwerrboundA = (3 + 16ε)ε` (ε = 2⁻⁵³) settles
  every row where some orientation is certainly non-zero;
- the few rows where the filter cannot rule out collinearity are re-checked exactly, in rational arithmetic
  (`fractions.Fraction`) on the original float coordinates.

"Exactly 0.0 when exactly collinear" (D12) then holds geometrically, and fewer cases depend on the R3 bound (Task 2).

**If declined (not taken):** the owner names a different rule; Task 1's `test_team_shape_matches_legacy`, Task 2's
`test_hull_area_precision_flat_within_abs_bound`, and the ADR/CHANGELOG hull sentence (Tasks 24, 28) follow it.

### R4 — Interim Tier-B base values for commit 1

**Finding.** §7.14 fixes a default only for `butterworth_cutoff_hz` (0.4 Hz). Commit 1 needs a base value for every
Tier-B field. Commit 2's D1-generated per-provider values replace them (§7.14: commit 1 is never released alone). The
base values also serve any provider absent from the generated map.

**Recommendation** (each value is derived from a stated rule, never guessed):

| Field | Interim base | Rule |
|---|---|---|
| `butterworth_cutoff_hz` | 0.4 | §7.14 (Moura 2013/2016) |
| `band_low_cpm`, `band_high_cpm` | 0.22, 0.83 | Moura 2013's reported median-frequency range |
| `welch_segment_s` | 400.0 | the §8.2 rule applied to the interim band on a 2,700 s half: resolution ≤ 0.61/4 cycles·min⁻¹ needs ≥ 393.4 s; 400 s gives 12 segments ≥ 8. Task 20 pins `welch_segment_rule(0.22, 0.83, 2700.0) == 400.0` |
| `min_shift_s` (every signal) | 60.0 | 4 × `xcorr_max_lag_s`: a shift at least four times the longest lag any method treats as coupling (the 4 × L rule of §7.8.2) |
| `vc_epsilon` (every signal) | 0.0 | no stationary drop until the noise floor is measured (Moura 2016 drops nothing) |
| `possession_gap_s` | 2.0 | bridges the flight of a typical pass, so a completed pass does not split a spell |
| `max_detection_gap_s` | 0.5 | the repo's existing interpolation bridge, `PreprocessConfig.max_gap_seconds` |
| `min_observed_fraction` (every family) | 0.5 | half the on-pitch players detected; fully observed providers are unaffected (their fraction is 1.0) |

**Ruling: APPROVED with amendment A3.** Three of these rules are reasoning rather than evidence: the 2.0 s possession
gap, the 0.5 minimum observed fraction and the 60 s minimum shift. They are principled placeholders, not measured
values. A3 ensures no placeholder reaches a release:
- **One source for Tier-B bases.** They are single-sourced in the generated file,
  `_provider_params_generated.BASE_COORDINATION_PARAMS`, stamped with `BASE_SOURCE ∈ {"interim", "derivation"}`.
  `CoordinationParams`' Tier-B field defaults read from it, so there is no second copy of the values.
- **Commit 1.** The codegen's empty-input render writes the table above as `INTERIM_BASE`, with
  `BASE_SOURCE = "interim"` (Task 19).
- **Commit 2.** D1 also derives **pooled, provider-neutral** values by the same rules (Task 20):
  - each provider carries equal weight;
  - within a provider, each match carries equal weight.

  The base's intended population is "a provider not in the map". The rationale is in Task 20's Reduce bullet, which
  answers TF58-PLAN-02. D2's moved selections apply to these values as to the per-provider ones (Task 22). The
  regenerated file
  carries `BASE_SOURCE = "derivation"`, so the base defaults, which also serve any provider absent from the map, are
  derived.
- **Gate.** Task 28 adds `test_base_defaults_are_derived`, which fails if `BASE_SOURCE == "interim"`. A release can
  therefore never ship the placeholders.

**If declined (not taken):** the owner's values replace `CoordinationParams`' interim bases (Task 14) and Task 20's
`welch_segment_rule` pin.

### R5 — `coord_rsi_switch_rate_per_min` has no unit in the closed `Unit` vocabulary

**Finding.** §9.4 amends ADR-048 with one new unit, `"cycles/min"`, for the median and peak frequencies. The RSI
sign-switch rate (§7.8.7, "sign-switch rate per minute") is also a per-minute rate. `feature_glossary.Unit` has
`passes/min` and `actions/min`, and neither is correct for it. A switch is half a cycle, so `cycles/min` would be
wrong by a factor of 2.

**Recommendation.** Add `"switches/min"` to `Unit` beside `"cycles/min"`, recorded in the same ADR-048 amendment.

**Ruling: APPROVED.**

**If declined (not taken):** the owner picks an existing unit, or redefines the column as full sign-change cycles per minute
(half the value, unit `cycles/min`). Task 18 Step 9 follows.

---

## Plan-level concretisations (no ruling needed; listed so the reviewer can check each)

- **C1 — Byte-identity mechanism.** `compute_defensive_line` must stay byte-identical (§7.5 D13), and
  `compute_team_shape` must keep every column except the hull byte-identical (§9.2). A probe on this machine
  (scratchpad `tie_probe.py`, not committed) found two facts:
  - numpy's default `argsort` is **not** stable for small arrays with ties: 21,312 of 40,000 tie-heavy arrays sorted
    differently from `kind="stable"`. numpy dispatches a SIMD sort.
  - The committed fixtures contain exact x-ties at the back-line cut: 100 groups in `j03wmx_slice`, 9 in
    `idsse_oldpath_harness_golden`, 1 each in the metrica and skillcorner slim fixtures.

  numpy also sums 8 or more values with 8-way pairwise accumulation. The kernels therefore reduce **count-bucketed
  compact rows**: the rows of each bucket have exactly `n` valid values, left-aligned in the legacy loop's row order.
  numpy then applies the same 1-D routine per row, so the pairwise-sum order and the default-`argsort` tie order are
  both identical to the per-group loop on every platform. A trailing-zero-padded row would not be.
- **C2 — Array goal-relative transforms.** "array-applied" (§7.10) is implemented as `to_goal_relative_x_array` /
  `to_goal_relative_y_array` in `tracking/_geometry.py`, beside the scalar functions. Both reuse `_flip` and the pitch
  constants, and are parity-tested element-wise against the scalar forms. `GEOMETRY_VERSION` does not change: no
  number moves.
- **C3 — The stale-artifact detector covers D1/D2 outputs.** `tests/scripts/test_input_contracts.py::_artifacts_for`
  scans only `metrics.json`. D1 writes `derivation.json` and D2 writes `calibration.json` (§8.2, §8.4). The detector is
  widened to any `*.json` whose top level carries `input_contract.driver`, with plants on both sides, so §8.3's
  `declare_inputs` is effective for all three drivers.
- **C4 — `objective_id` on a dirty tree.** The D21 helper returns `"<qualname>@<commit>:<digest>"` (§12). When
  `git_provenance()` reports the tree `dirty` or `unknown`, it appends `+dirty-<uuid4 hex>`. A dirty or unknown tree
  therefore never resumes a store across runs, which is the fail-closed reading of §12. Crash recovery within one
  clean-tree run is unchanged.
- **C5 — `n_phases` sample assignment.** The window's `m` samples are ranked `i = 0..m−1`. Sample `i` goes to phase `k`
  when `(i+1)/m ∈ ((k−1)/n, k/n]` (§7.6). This rank form puts the first sample in phase 1. A time-fraction form would
  leave it unassigned.
- **C6 — Minimum samples for relative-phase window statistics:** 3 valid samples per window and per subdivision. This
  is the vector-coding minimum of §7.8.3, reused so the two phase-based families degrade identically. A window with
  fewer samples is `too_short`.
- **C7 — Key tokens.**
  - `axis` ∈ {`x`, `y`, `scalar`}. Signals ending in `_x`, plus `team_length`, `player_x`, `defensive_line_x`,
    `compactness_x` and `back_line_high_x`, are `x`. Their `_y` counterparts, plus `team_width` and `player_y`, are
    `y`. `stretch_index`, `spread`, `convex_hull_area` and `possession` are `scalar`.
  - `window_kind` and `window_source` are exactly the §7.6 sets.
- **C8 — Team ids on the team-pair tables.** `COORD_TEAM_SYNC` and `COORD_RSI` carry `team_a_id` and `team_b_id` as
  non-key columns. RSI = SI_A − SI_B, so its sign is meaningless without them.
- **C9 — `COORD_PAIR_PHASE` columns.** It carries every relative-phase column except the two phase-validity fractions
  (phase validity is a per-series property) and every vector-coding column, in both cases without surrogates (§7.12).
- **C10 — Private imports.** `coordination` imports three private modules, each allowlisted with its reason in
  `tests/coordination/test_import_allowlist.py` and recorded in `docs/PRIVATE_CONSUMERS.md`:
  - `tracking._provider_visibility` (`dead_ball_observed`, `assert_detection_aware_visibility`,
    `_DETECTION_AWARE_PROVIDERS`);
  - `tracking._geometry` (C2);
  - `silly_kicks._frame_index` (not tracking-private, recorded anyway).

  This follows §7.1 as approved. `PRIVATE_CONSUMERS.md` row 47 says "promote to `tracking.__all__` only if a
  cross-package consumer appears". `coordination` is that consumer, and the promotion stays an owner option. The plan
  does not take it, because the spec did not.
- **C11 — Surrogate acceleration scope.**
  - The FFT circular cross-correlation identity (§7.9) is exact only when a window covers its whole segment, so it is
    used exactly there.
  - Sub-segment windows (possession, sliding) use the direct vectorised gather. It computes the same estimator and
    costs O(window length × K).
  - Both paths are parity-tested against each other.
- **C12 — Two keyword-only overrides on `build_coordination_signals`.** They are needed by approved legs, and both
  default to the spec's behaviour:
  - `stoppage_evidence: Literal["auto", "ball_state", "events", "none"] = "auto"`: the D20 leg (§8.5) must compute each
    metric under ball-state, event and no splitting on GS/IDSSE.
  - `detection: Literal["auto", "detection_aware", "fully_observed"] = "auto"`: the D9 occlusion leg (§8.2) must run
    detection-aware handling on occluded fully observed frames.

  `"auto"` is §7.6 / §7.11 exactly.
- **C13 — Dyad rows have no vector-coding or coherence method.** On a `dyad` row every `coord_vc_*` and `coord_coh_*`
  column, **and** `coord_vc_source` / `coord_coh_source`, is `<NA>`. The reason is structural, derivable from `level`,
  and asserted by `test_dyad_rows_have_null_vc_and_coh_by_level`. Every other NaN carries a token.
- **C14 — `build_selection_artifact` is carrier-specific.** It hard-codes `beta`/`gamma` (`calibration/_selection.py:131`).
  §8.4 names it for D2's output, but D2 cannot call it. D2 uses `select_recommended_point` unchanged and writes its own
  `calibration.json` with the same provenance fields (`moved`, `reason`, `run_commit`, `run_tree_dirty`). The public
  builder is untouched (Hyrum's law: `carrier_selected.json` has consumers).
- **C15 — §7.16 re-sweep additions.** At plan time, `tests/calibration/test_spaces.py` calls the builders 7 times
  (lines 8, 20, 26, 35, 36 and the two `xt_bandwidth_config` calls from line 39). The spec lists `:8` only. Task 21
  migrates every call; the sweep is the floor.
- **C16 — `CoordinationParams` map fields.** `vc_epsilon`, `min_shift_s`, `min_observed_fraction` and
  `include_goalkeeper` are complete read-only maps: every key is present, and `__post_init__` asserts the exact key set.
  `for_provider` merges overrides key-wise. The class defines `__hash__` over a canonical tuple so it stays hashable.

- **C17 — Three ruthless pins, not two.** `pyproject.toml` pins `ruthless-efficiency[optuna]>=0.6.0` in
  `[calibration]` (l.86), `[test]` (l.127) and `[train]` (l.138); §10 names two. CI installs `[test]`, so all three
  are set to exactly `>=0.7.0` (Task 21). That removes any `<0.7.0` ceiling the F1b cycle's CI hotfix may have added,
  and the removal ships in the same commit as the full `objective_id` migration (Task 21 Step 4, Task 26 Step 2).
- **C18 — Residual analysis at low native rates.**
  - `residual_analysis_cutoff` evaluates only the grid frequencies whose Winter-corrected design frequency is below
    Nyquist. At 10 Hz, §8.2's 0.1–5.0 Hz grid ends near 4.29 Hz.
  - The "linear high-frequency tail" is the upper half of the evaluated grid (Task 4).
- **C19 — Spec §9.1's 225° example.** The printed Moura Eq. 2 gives 45° for a 225° coupling. That is the wrong angle
  but the same in-phase class. The misclassification happens at 135° and 315° (anti-phase read as in-phase). The
  regression pin asserts all three (Task 10).
- **C20 — Team sync uses the segment-level cluster amplitude.** φ̄_k is taken over the continuous segment, the
  `cluster_amplitude` series. §7.9's team-sync surrogate shifts that series "within each continuous segment", which
  presupposes a segment-level series (Task 17).
- **C21 — Window-builder edges.**
  - Sliding windows are full-length only.
  - A possession with no next possession in its period ends at the period's last frame, with `<NA>` terminals
    (Task 15).
- **C22 — Id columns.**
  - Pair keys (`team_a_id`, `team_b_id`, `player_a_id`, `player_b_id`) are canonical ids (`object`).
  - Single-entity ids (`game_id`, `team_id`, `player_id`) are `restore_id_dtype`-restored.
  - All ids are declared `object`, as in the `TERRITORY_COLUMNS` precedent (Task 14).
- **C23 — `links` reuse in `possession_windows_from_actions`.** A linked action's start and end times are its linked
  frame's `time_seconds` (Task 15).
- **C24 — Orientation is applied algebraically per window,** from team A's goal-relative frame. A point reflection
  negates a centred positional signal: its Hilbert phase shifts by exactly π, a Pearson r that involves exactly one
  positional signal flips sign, and its differences are negated. This keeps D15's "phases once per segment"
  compatible with §7.10's per-window reference team (Task 16, Task 17).
- **C25 — Window-source mix.**
  - `period` windows may accompany one possession source.
  - `possession_events` and `possession_tracking` together are refused.
  - `caller` windows mix with nothing.

  This reconciles §7.3's refusal with §7.2's orchestrator default (Task 15).
- **C26 — D2 levels are relative to each provider's Tier-B value.** Multipliers, or additive offsets for
  `min_observed_fraction`, so one grid and one OAT baseline serve every provider (Task 22).
- **C27 — The D2 `objective_id` spans every shard generation it reads.** It is a digest of the sorted generation
  tokens, so a single regenerated pass changes it (Task 22).

---

## Global Constraints

- **Scope.** Everything in §6 "In scope" and every §9 test. Nothing is deferred, dropped or simplified without the
  owner's explicit approval. If a step cannot be done as written, stop and ask; never park it on `TODO.md`.
- **Commits.** Exactly two commits land on `feat/tf58-team-coordination` (§13): commit 1 (Task 25) and commit 2
  (Task 28). Outside those two approval stops, no step runs `git commit`, `git push` or `git stash`, with exactly two
  named exceptions. Each exception is its own outward action behind its own explicit owner approval, requested at the
  step, and neither adds a commit to the feature branch:
  - **(a) Task 26 Step 7.** The rebase force-push `git push --force-with-lease`, needed only if commit 1 was already
    pushed when F1b or DAS merges.
  - **(b) Task 28 Step 5.** The ADR-074 `.test_durations` capture. It runs on a **throwaway** branch cut from the
    feature branch; one commit there adds the temporary `durations-capture` job. It is pushed, a draft PR to `main` is
    opened (CI runs only on `pull_request` to `main` or a push to `main`, per `ci.yml:3-7`), and the PR is closed and
    the branch deleted once the artifact is downloaded. The throwaway commit never reaches the feature branch.

  The approval request for commit 1 covers **commit + push + PR**. The request for commit 2 covers **commit + push**.
  Never create the `~/.claude-git-approval` sentinel yourself. The trailer is the one mandated for the implementing
  session.
- **Branch.** `feat/tf58-team-coordination` only. No worktree.
- **Clones.** Work only in `D:\Development\karstenskyt__silly-kicks_ragnarok`. Never read-write the sibling clones
  (`karstenskyt__silly-kicks` = F1b, `_part-deux` = native DAS). Never install into their `.venv`. The ruthless repo is
  read-only to this session except for an editable install **into ragnarok's own `.venv`** (Task 21).
- **Version / ADR / PR-S numbers** are assigned at commit-prep only (§16). Until then the ADR file is
  `docs/superpowers/adrs/ADR-111-team-coordination.md`.
- **Numerics.** float64 inside every kernel. Coordinates are read with `.to_numpy(dtype="float64")` (ADR-106
  boundary). TF-58 never writes coordinates back into frame storage.
- **Dependencies.** No new runtime dependency. `numba` is optional (existing `[numba]` extra), and results never depend
  on whether it is installed.
- **Repo conventions** (AGENTS.md):
  - no `apply(axis=1)`;
  - every `warnings.warn(..., stacklevel=2)`;
  - dtype-safe ids through `silly_kicks.id_compat`;
  - no rescan-in-loop (`group_rows`);
  - direction only from the `GoalMap`;
  - `observed=True` on every categorical `groupby`;
  - every public function, class and method carries an Examples section.
- **Per-task quality bar.** Run on the files the task touches:
  - `.venv\Scripts\python -m ruff check <files>`;
  - `.venv\Scripts\python -m ruff format --check <files>`;
  - `py -3.14 -m pyright <files>`;
  - the task's tests.

  A task is done only when all four are clean.
- **Full gate** (Task 25 and Task 28) at CI scope:
  - `.venv\Scripts\python -m pytest tests/ -m "not e2e" --benchmark-skip -q`;
  - the same suite on a pandas-2 interpreter (Task 0 Step 3 recipe);
  - `.venv\Scripts\python -m ruff check silly_kicks/ tests/ scripts/`;
  - `.venv\Scripts\python -m ruff format --check silly_kicks/ tests/ scripts/`;
  - bare `py -3.14 -m pyright`;
  - `.venv\Scripts\python -m pytest --doctest-modules silly_kicks/ --ignore-glob="*/_[!_]*.py" -q`.
- **Evidence discipline.** Claims about a gate quote its assertion body (AGENTS.md §Testing). Every band is tested from
  both sides. Every counterfactual asserts it moved something (non-vacuity). Detection lands before the fix
  (ADR-051).

---

## File map

**Created (package):**

| File | Responsibility |
|---|---|
| `silly_kicks/coordination/__init__.py` | public surface, exact `__all__` (§7.2) |
| `silly_kicks/coordination/_columns.py` | every schema dict, key tuple, vocabulary, `SignalSpec`, `COORD_SIGNALS` |
| `silly_kicks/coordination/_config.py` | `CoordinationParams` |
| `silly_kicks/coordination/_provider_params_generated.py` | GENERATED map; empty in commit 1 |
| `silly_kicks/coordination/_report.py` | `CoordinationReport`, `CoordinationCoverageWarning` |
| `silly_kicks/coordination/_windows.py` | window builders, contract validation, stoppage evidence, phase assignment |
| `silly_kicks/coordination/_catalog.py` | `PairSpec`, default catalog, methods per level |
| `silly_kicks/coordination/_signals.py` | `build_coordination_signals`, `CoordinationSignals`, `PeriodSignals`, `PlayerSeries` |
| `silly_kicks/coordination/_compute.py` | the seven family computes, `compute_team_coordination`, `CoordinationResult` |
| `silly_kicks/coordination/_series.py` | `compute_coordination_series` |
| `silly_kicks/coordination/_kernels/__init__.py` | empty |
| `silly_kicks/coordination/_kernels/_circular.py` | circular summaries, 12-bin histogram, near-in-phase |
| `silly_kicks/coordination/_kernels/_phase.py` | centring, reflect padding, analytic phase, phasors, validity |
| `silly_kicks/coordination/_kernels/_xcorr.py` | lagged Pearson, Fisher-z pooling, summary |
| `silly_kicks/coordination/_kernels/_vector_coding.py` | coupling angle, Table 1 classification, stationarity |
| `silly_kicks/coordination/_kernels/_spectral.py` | median frequency, Welch spectra, pooled coherence |
| `silly_kicks/coordination/_kernels/_cluster.py` | cluster phase and ρ statistics |
| `silly_kicks/coordination/_kernels/_entropy.py` | SampEn / Cross-SampEn, exact counters |
| `silly_kicks/coordination/_kernels/_surrogates.py` | seeded time-shift / IAAFT, accelerated statistics, rank percentile |
| `silly_kicks/coordination/_kernels/_numba.py` | optional `@njit` inner loops |

**Created (seams, scripts, tests, docs):**
- `silly_kicks/tracking/_collective.py`
- `silly_kicks/tracking/preprocess/_butterworth.py`
- scripts:
  - `scripts/_reliability.py`
  - `scripts/_corpus_visibility.py`
  - `scripts/_coordination_corpus.py` (shared D1–D3 per-match plumbing)
  - `scripts/_coordination_occlusion.py`
  - `scripts/_coordination_params_codegen.py`
  - `scripts/_coordination_thresholds.py`
  - `scripts/_coordination_hypotheses.py`
  - `scripts/derive_coordination_params.py`
  - `scripts/calibrate_coordination.py`
  - `scripts/validate_team_coordination.py`
- tests:
  - `tests/tracking/_legacy_collective_oracle.py`
  - `tests/tracking/test_collective_parity.py`
  - `tests/tracking/test_collective_kernels.py`
  - `tests/tracking/test_collective_variables.py`
  - `tests/tracking/test_collective_delegation.py`
  - `tests/tracking/test_preprocess_butterworth.py`
  - `tests/tracking/test_dead_ball_taxonomy.py`
  - `tests/tracking/test_geometry_array_twins.py`
  - `tests/coordination/` (tree in Tasks 8–18)
  - `tests/scripts/test_reliability_module.py`
  - `tests/scripts/test_corpus_visibility_module.py`
  - `tests/scripts/test_provenance_objective_id.py`
  - `tests/scripts/test_coordination_driver_modules.py`
  - `tests/coordination/test_provider_params_generated.py` (commit 2)
  - `tests/scripts/test_derive_coordination_params.py`
  - `tests/scripts/test_calibrate_coordination.py`
  - `tests/scripts/test_validate_team_coordination.py`
- `docs/superpowers/adrs/ADR-111-team-coordination.md` (renamed at commit-prep)
- R2 fixture: `tests/datasets/tracking/idsse_half/{frames.parquet, generate_fixture.py, README.md}`

**Modified:**
- tracking seams:
  - `silly_kicks/tracking/_team_shape.py`
  - `silly_kicks/tracking/_defensive_line.py`
  - `silly_kicks/tracking/_provider_visibility.py`
  - `silly_kicks/tracking/_geometry.py`
  - `silly_kicks/tracking/__init__.py`
- preprocess:
  - `silly_kicks/tracking/preprocess/{__init__.py, _config_dataclass.py, _smoothing.py}`
- `silly_kicks/metric_contracts.py`
- `silly_kicks/feature_glossary.py`
- `silly_kicks/calibration/_spaces.py`
- scripts:
  - `scripts/_provenance.py`
  - `scripts/validate_team_kpi_reliability.py`
  - `scripts/validate_gk_decision.py`
  - `scripts/train_ghost_gk.py`
  - `scripts/train_xshot_occurrence.py`
  - `scripts/train_xcross_attempt.py`
  - `scripts/calibrate_tracking_defaults.py`
  - `scripts/calibrate_xt_bandwidth.py`
  - `scripts/check_stage1_argmax.py`
- tests:
  - `tests/test_metric_contracts.py`
  - `tests/test_public_api_examples.py`
  - `tests/_scale_guarded.py`
  - `tests/test_scale_guards.py`
  - `tests/invariants/glossary_emitted_columns.py`
  - `tests/invariants/test_glossary_emitted_columns.py`
  - `tests/scripts/test_provenance_wiring.py`
  - `tests/scripts/test_input_contracts.py`
  - `tests/calibration/test_spaces.py`
  - `tests/tracking/test_xshot_occurrence_integration.py`
- config:
  - `.github/workflows/ci.yml` (numba cache key, both jobs)
  - `pyproject.toml`
  - `uv.lock`
- documentation:
  - `NOTICE`
  - `docs/c4/architecture.dsl` and `architecture.html`
  - `AGENTS.md`
  - `docs/context/tracking-metrics.md`
  - `docs/PRIVATE_CONSUMERS.md`
- commit 2:
  - `CHANGELOG.md`
  - `TODO.md`
  - `silly_kicks/_version.py`
  - `.test_durations`

---

## Task order and dependency direction

Pure cores first, then adapters (the dependency direction), so no step tests through a mock of something unwritten:

1. **Seams** (Tasks 1–7): the legacy oracle, the collective kernel, delegation, preprocess, taxonomy and geometry
   twins, the metric-contract re-key, the script helpers.
2. **Coordination kernels** (Tasks 8–13): numpy/scipy only, zero mocks.
3. **Package plumbing** (Tasks 14–17): columns, config and report; windows; catalog and signals; computes.
4. **Package gates** (Task 18).
5. **Drivers** (Tasks 19–23; Tasks 21–22 need ruthless 0.7.0).
6. **Documentation and commit 1** (Tasks 24–25), then the rebase (Task 26, whenever F1b/DAS merge).
7. **DGX runs and commit 2** (Tasks 27–28).

---

### Task 0: Baseline, environments, sweep

**Files:** none in the repo. Logs go to the session scratchpad, never into the tree.

- [ ] **Step 1: Pin the tree.**
  - `git branch --show-current` → `feat/tf58-team-coordination`.
  - `git log -1 --oneline` → `05cfa56` (or the post-rebase `main` tip, Task 26).
  - `git status --short` shows only `?? docs/superpowers/specs/2026-09-26-tf58-team-coordination-design.md` and
    `?? docs/superpowers/plans/2026-09-26-tf58-team-coordination.md`.
- [ ] **Step 2: Pandas-3 baseline.**
  - Command: `.venv/Scripts/python -m pytest tests/ -m "not e2e" -p no:randomly --benchmark-skip -q`. It takes
    longer than 30 s, so run it in the background.
  - Record in the scratch log: passed / failed / skipped counts, wall time, and the node id of every pre-existing
    failure.
  - Every later gate compares against this list. A pre-existing failure is context; a new one is yours.
- [ ] **Step 3: The pandas-2 leg recipe.** This mirrors CI's ubuntu-3.10 leg, which resolves pandas 2 because pandas 3
  needs Python ≥ 3.11. It leaves no file in the tree:
  ```
  uv run --no-project --python 3.10 --with-editable ".[kloppy,xgboost,das,test]" python -m pytest tests/ -m "not e2e" -p no:randomly --benchmark-skip -q
  ```
  - Confirm `uv run --no-project --python 3.10 --with-editable ".[kloppy,xgboost,das,test]" python -c "import pandas; print(pandas.__version__)"` prints `2.x`.
  - Record the baseline as in Step 2.
  - Between Task 21's floor bump and the 0.7.0 PyPI release, add
    `--with-editable "D:/Development/karstenskyt__ruthless-efficiency[optuna]"`.
- [ ] **Step 4: Pre-change timings.** A scratchpad script, never committed, times `compute_team_shape` (per frame per
  team) and `compute_defensive_line` (per frame-team) on `tests/datasets/elastic_sync/j03wmx_slice/frames.parquet`.
  Record the numbers for the ADR's performance section. Task 3 Step 8 re-measures.
- [ ] **Step 5: Re-run the §7.16 caller sweep.**
  ```
  git grep -n "compute_team_shape(" -- silly_kicks scripts tests
  git grep -n "compute_defensive_line(" -- silly_kicks scripts tests
  git grep -n "smooth_frames(" -- silly_kicks scripts
  git grep -n "PreprocessConfig(" -- silly_kicks scripts tests
  git grep -n -E "icc1|split_half_reliability|type_ii_slope|compare_providers" -- scripts tests
  git grep -n "validate_corpus_visibility" -- scripts tests
  git grep -n -E "stage1_config|stage2_config|xt_bandwidth_config" -- silly_kicks scripts tests
  git grep -n "StoreConfig(" -- silly_kicks scripts tests
  git grep -n "ruthless-efficiency" -- pyproject.toml
  ```
  Compare each hit with §7.16 and with C15/C17. A hit that neither lists is reported to the owner before Task 1.
  Plan-time result, already accounted for:
  - `tests/calibration/test_spaces.py` has 7 builder calls (C15);
  - `pyproject.toml` pins ruthless in three extras — `[calibration]` l.86, `[test]` l.127, `[train]` l.138 —
    while §10 names two (**C17**: all three are bumped; CI installs `[test]`, so leaving it at `>=0.6.0` would let CI
    resolve a ruthless without `objective_id`).

---

### Task 1: Freeze the legacy collective oracle (detection before the change)

**Files:**
- Create: `tests/tracking/_legacy_collective_oracle.py`
- Create: `tests/tracking/test_collective_parity.py`

**Interfaces:**
- Produces:
  - `legacy_compute_team_shape(frames, team_id, *, n_defensive_lines=3) -> pd.DataFrame`
  - `legacy_compute_defensive_line(frames, *, goal_map, n=4, adaptive_max_n=5) -> pd.DataFrame`
  - `_legacy_select_n(xs_sorted, n, adaptive_max_n, p) -> int`

  Byte-for-byte copies of the functions at `05cfa56`.

- [ ] **Step 1: Write the oracle.**
  - Paste the bodies from `git show 05cfa56:silly_kicks/tracking/_team_shape.py` (lines 20–170: `_RESULT_COLS`,
    `compute_team_shape`) and `git show 05cfa56:silly_kicks/tracking/_defensive_line.py` (lines 106–361:
    `compute_defensive_line`, `_select_n`).
  - Rename the three functions with the `legacy_` / `_legacy_` prefix. Change nothing else except the internal
    `_select_n` call.
  - Keep the original imports (`numpy`, `pandas`, `scipy.cluster.hierarchy.fcluster/linkage`,
    `scipy.spatial.ConvexHull/QhullError`, `silly_kicks.id_compat.ids_match`,
    `silly_kicks.tracking._gk_resolve.GoalEndUnresolvedError/GoalMap`).
  - Module docstring: `"Frozen verbatim copy of compute_team_shape / compute_defensive_line / _select_n at 05cfa56 —
    the ADR D12/D13 parity oracle. Never edit except an import path."`
- [ ] **Step 2: Write the parity harness.** In `tests/tracking/test_collective_parity.py`:

```python
"""Parity gates for the vectorised collective kernel (TF-58 D12/D13, spec §7.5, §9.2).

compute_defensive_line must stay byte-identical; compute_team_shape must stay byte-identical except
convex_hull_area (<= 1e-9 relative where Qhull succeeds; <= 1e-9 m^2 absolute where Qhull raised on a
precision-flat, not exactly collinear, frame -- owner ruling R3). The oracle is the frozen 05cfa56 copy."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import compute_defensive_line, compute_team_shape, resolve_defended_goals
from tests.tracking._legacy_collective_oracle import legacy_compute_defensive_line, legacy_compute_team_shape

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FRAME_FIXTURES = (
    "tests/datasets/elastic_sync/j03wmx_slice/frames.parquet",
    "tests/datasets/tracking/action_context_slim/sportec_slim.parquet",
    "tests/datasets/tracking/action_context_slim/metrica_slim.parquet",
    "tests/datasets/tracking/action_context_slim/skillcorner_slim.parquet",
    "tests/datasets/sportec/idsse_slice/idsse_oldpath_harness_golden.parquet",
    "tests/datasets/tracking/synthetic/brief_outfielder.parquet",
    "tests/datasets/tracking/synthetic/gk_substitution.parquet",
    "tests/datasets/tracking/synthetic/sweeper_keeper.parquet",
)
_DL_VARIANTS = [(3, 5), (4, 5), (5, 5), ("adaptive", 3), ("adaptive", 4), ("adaptive", 5)]


def _frames(rel: str) -> pd.DataFrame:
    return pd.read_parquet(_ROOT / rel)


def _outcome(fn, *args, **kwargs):
    """(result, None) or (None, (exception type, message)) -- parity covers the raise path too."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001 -- parity of ANY raise is the assertion
        return None, (type(e), str(e))


@pytest.mark.parametrize("rel", _FRAME_FIXTURES)
@pytest.mark.parametrize(("n", "amn"), _DL_VARIANTS)
def test_defensive_line_byte_identical_to_legacy(rel, n, amn):
    frames = _frames(rel)
    goal_map = resolve_defended_goals(frames)
    new, new_err = _outcome(compute_defensive_line, frames, goal_map=goal_map, n=n, adaptive_max_n=amn)
    old, old_err = _outcome(legacy_compute_defensive_line, frames, goal_map=goal_map, n=n, adaptive_max_n=amn)
    assert new_err == old_err
    if old is not None:
        pd.testing.assert_frame_equal(new, old, check_exact=True, check_dtype=True)
```

  - Add `test_team_shape_matches_legacy[rel]`. It runs every non-NA team id (`team_id` values of non-ball rows) of
    every fixture through both functions and compares:
    - every column except `convex_hull_area` with `assert_frame_equal(check_exact=True, check_dtype=True)`;
    - `convex_hull_area` elementwise: exact equality where the legacy value is `0.0` **and** the frame is exactly
      collinear; `abs(new - old) <= 1e-9` where legacy is `0.0` from a precision-flat `QhullError`; otherwise
      `np.testing.assert_allclose(rtol=1e-9, atol=0)`.
    - The test classifies each frame by recomputing `ConvexHull` itself, so the three branches are data-driven.
  - Add two precondition tests. They make the harness non-vacuous (ADR-032, C1):
    - `test_parity_fixtures_contain_cut_ties`: over all fixtures, count the `(game, period, frame, team)` outfield
      groups where the sorted x (both directions) ties exactly at position `n−1`/`n` for some `n ∈ {3,4,5}`. Assert
      `≥ 50` for `j03wmx_slice` alone (measured at plan time: 100). This proves the default-`argsort` tie path is
      exercised.
    - `test_parity_fixtures_contain_rtl_small_and_adaptive_groups`: assert the fixture union contains groups with
      `team_attacking_direction == "rtl"`, groups with fewer than 3 outfield players, and groups with at least 6
      (where the adaptive `n = 5` cut exists).
  - If a fixture lacks a column either function requires, the harness fails loudly (never skips). Fix the fixture
    list, not the assertion.
- [ ] **Step 3: Run.** `.venv/Scripts/python -m pytest tests/tracking/test_collective_parity.py -q -p no:randomly`
  → **PASS** on unchanged production code. This proves the oracle is a faithful copy. The test file is unchanged
  from here on; Task 3 turns it into the regression gate.
- [ ] **Step 4:** ruff + format + pyright on both files.

---

### Task 2: Collective-variable array kernels (`tracking/_collective.py`)

**Files:**
- Create: `silly_kicks/tracking/_collective.py`
- Test: `tests/tracking/test_collective_kernels.py`

**Interfaces — produces (consumed by Tasks 3 and 16):**

```python
COLLECTIVE_VARIABLES: tuple[str, ...] = (
    "centroid_x", "centroid_y", "team_length", "team_width", "stretch_index",
    "stretch_x", "stretch_y", "spread", "convex_hull_area",
)
BACK_LINE_VARIABLES: tuple[str, ...] = (
    "defensive_line_x", "back_line_high_x", "compactness_x", "lateral_width", "max_lateral_gap", "back_n_count",
)

def pack_groups(codes: np.ndarray, x: np.ndarray, y: np.ndarray, n_groups: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    # -> (pos (G, P, 2) float64 NaN-padded, valid values LEFT-ALIGNED in input row order;
    #     counts (G,) int64; first_row (G,) int64 = positional index of each group's first input row)
def compact_rows(pos: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    # (T, P, 2) + (T, P) bool -> left-aligned (T, P, 2) (slot order preserved) + counts (T,)
def hull_area_batch(pos: np.ndarray, counts: np.ndarray) -> np.ndarray: ...
    # NaN where count < 3; exactly 0.0 where every orientation determinant is exactly 0; else the area
def collective_from_positions(pos: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]: ...
    # keys == COLLECTIVE_VARIABLES; NaN below each variable's minimum n (§7.5 table)
def back_line_batch(pos: np.ndarray, counts: np.ndarray, defends_x0: np.ndarray, *,
                    n: int | Literal["adaptive"], adaptive_max_n: int) -> dict[str, np.ndarray]: ...
    # keys == BACK_LINE_VARIABLES + ("valid",); back_n_count int64, meaningful only where valid (count >= 3)
```

- [ ] **Step 1: Write the failing tests.**

```python
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from silly_kicks.tracking._collective import hull_area_batch


def _qhull(p: np.ndarray) -> tuple[float, bool]:
    try:
        return float(ConvexHull(p).volume), False
    except QhullError:
        return 0.0, True


def test_hull_area_matches_qhull_random():
    rng = np.random.default_rng(58)
    for n in range(3, 15):
        pts = rng.uniform((0.0, 0.0), (105.0, 68.0), size=(400, n, 2))
        got = hull_area_batch(pts, np.full(400, n, dtype=np.int64))
        want = np.array([_qhull(p)[0] for p in pts])
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=0)


def test_hull_area_exact_collinear_and_coincident_is_zero():
    line = np.array([[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [5.0, 10.0]]])
    same = np.array([[[3.0, 3.0]] * 4])
    assert hull_area_batch(line, np.array([4]))[0] == 0.0
    assert hull_area_batch(same, np.array([4]))[0] == 0.0


def test_hull_area_precision_flat_within_abs_bound():   # owner ruling R3
    rng = np.random.default_rng(7)
    raised = 0
    for _ in range(200):
        x = np.sort(rng.uniform(0, 100, 6))
        pts = np.column_stack([x, 0.3 * x + 1.0 + rng.normal(0, 1e-13, 6)])[None]
        want, flat = _qhull(pts[0])
        got = hull_area_batch(pts, np.array([6]))[0]
        if flat:
            raised += 1
            assert abs(got) <= 1e-9
        else:
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=0)
    assert raised > 0, "no precision-flat case reached Qhull's error path -- the R3 branch is untested"
```

  Also write:
  - `test_hull_area_adversarial` covers:
    - a hull vertex duplicated;
    - points on a hull edge;
    - exactly 3 points;
    - 11 points of which 7 are interior;
    - integer-grid points with collinear subsets.

    Each compares to `_qhull`, with `rtol=1e-9`.
  - `test_hull_area_fewer_than_three_is_nan` covers counts 0, 1 and 2.
  - A2 exact-predicate tests, against the exact rational oracle `_collinear_exact`:
    - `test_exact_predicate_rejects_float_false_positive` (non-vacuity for A2). Input: the points (0, 0),
      (1 + 2⁻⁵², 1) and (1 + 2⁻⁵¹, 1 + 2⁻⁵²). Assert first that the **naive float determinant is exactly `0.0`**, so a
      `det == 0.0` test would call them collinear. The exact determinant is −2⁻¹⁰⁴, and `_exactly_collinear` must
      return `False`.
    - `test_exact_predicate_accepts_exactly_collinear_dyadic_sets`: points `x = m·2⁻¹⁰`, `y = 3x + 7` (exactly
      representable), with a non-origin anchor, give `True`. So does a set whose points all coincide.
    - `test_exact_predicate_agrees_with_rational_oracle`: 2,000 seeded sets, `n ∈ 3..11`, mixing random,
      near-collinear (±1 ulp) and exactly collinear dyadic points; `_exactly_collinear` equals `_collinear_exact` on
      every row.
    - `test_exact_fallback_runs_only_on_unsettled_rows` (structural, `tests/_perf_structural.call_counter` on
      `_collinear_exact`): on 1,000 random well-spread frames the count is 0; on the near-collinear set it equals the
      number of rows the float filter cannot settle.
  - `test_spread_identity_matches_naive_double_sum` (random `n ∈ 2..14`): `spread == sqrt(Σ_{i<j} ‖p_i−p_j‖²)`
    within `rtol=1e-9`.
  - `test_collective_from_positions_bit_identical_to_per_row_numpy`. It uses random rows with `n` from 1 to 14,
    left-aligned with a NaN tail. For every row it compares, with `==`:
    - `centroid_x` against `np.mean(row_x)`, and likewise `centroid_y`;
    - `team_length`/`team_width` against `max − min`;
    - `stretch_index` against `np.mean(np.sqrt((xs−cx)**2+(ys−cy)**2))`;
    - `stretch_x`/`stretch_y` against `np.mean(np.abs(xs−cx))`.

    This is the C1 guarantee.
  - `test_back_line_batch_matches_legacy_rows`:
    - Input: 3,000 random rows, `p ∈ 1..11`, x drawn from the **integers 0..10** (tie-heavy), y from uniform, both
      `defends_x0` values.
    - Variants: every `(n, amn)` in `_DL_VARIANTS`.
    - The expected values come from the legacy per-row code, inlined in the test:
      - `order = np.argsort(xs)` or `np.argsort(-xs)`;
      - `n_eff = _legacy_select_n(...)`;
      - the six reductions exactly as in the oracle.
    - Compare with `np.testing.assert_array_equal`, where NaN equals NaN.
  - `test_pack_groups_preserves_within_group_order`: for shuffled codes, each group's left-aligned values equal the
    input rows of that code in input order; `first_row` points at each group's first input row.
  - `test_compact_rows_left_aligns_valid_slots`: valid slots keep their relative slot order, and counts equal
    `valid.sum(1)`.

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/tracking/test_collective_kernels.py -q` → **FAIL**
  (`ModuleNotFoundError: silly_kicks.tracking._collective`).
- [ ] **Step 3: Implement.** The C1 mechanism is the load-bearing detail:
  - every reduction runs on a **compact** `(R, n)` slice holding only rows whose count is exactly `n`;
  - numpy therefore applies its 1-D per-row routine: pairwise summation for `n ≥ 8`, and the platform's default
    `argsort` for ties;
  - that makes the results identical to the legacy per-group loop.

```python
def pack_groups(codes, x, y, n_groups):
    order = np.argsort(codes, kind="stable")          # stable: within-group input order preserved
    sc = codes[order]
    counts = np.bincount(codes, minlength=n_groups).astype(np.int64)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    rank = np.arange(sc.size, dtype=np.int64) - starts[sc]
    width = int(counts.max()) if n_groups else 0
    pos = np.full((n_groups, width, 2), np.nan)
    pos[sc, rank, 0] = x[order]
    pos[sc, rank, 1] = y[order]
    first_row = order[starts].astype(np.int64) if n_groups else np.empty(0, dtype=np.int64)
    return pos, counts, first_row


def compact_rows(pos, valid):
    order = np.argsort(~valid, axis=1, kind="stable")  # valid slots first, slot order kept
    return np.take_along_axis(pos, order[..., None], axis=1), valid.sum(axis=1).astype(np.int64)


def _hull_area_fixed_n(p):                          # (R, n, 2), n >= 3, all valid
    r_count, n, _ = p.shape
    d = p[:, None, :, :] - p[:, :, None, :]           # d[r, i, j] = p_j - p_i
    coincident = (d[..., 0] == 0.0) & (d[..., 1] == 0.0)   # includes j == i
    ang = np.where(coincident, np.inf, np.arctan2(d[..., 1], d[..., 0]))
    ang.sort(axis=2)                                  # coincident points carry no direction -> +inf, last
    m = n - coincident.sum(axis=2)                    # distinct directions seen from point i
    gaps = np.diff(ang, axis=2)
    k = np.arange(n - 1)
    gaps = np.where(k[None, None, :] < (m - 1)[..., None], gaps, -np.inf)
    last = np.take_along_axis(ang, np.maximum(m - 1, 0)[..., None], axis=2)[..., 0]
    wrap = np.where(m >= 1, ang[..., 0] + 2.0 * np.pi - last, 2.0 * np.pi)
    on_hull = np.maximum(gaps.max(axis=2, initial=-np.inf), wrap) >= np.pi
    # EXACT collinearity (A2) -> exactly 0.0 (the QhullError contract, decided geometrically, not by float == 0.0)
    collinear = _exactly_collinear(p)
    # shoelace over hull points ordered by angle around their mean (inside the hull); centred to limit cancellation
    cnt = on_hull.sum(axis=1)
    cx = np.where(on_hull, p[..., 0], 0.0).sum(axis=1) / cnt
    cy = np.where(on_hull, p[..., 1], 0.0).sum(axis=1) / cnt
    qx, qy = p[..., 0] - cx[:, None], p[..., 1] - cy[:, None]
    theta = np.where(on_hull, np.arctan2(qy, qx), np.inf)
    order = np.argsort(theta, axis=1, kind="stable")
    qx, qy = np.take_along_axis(qx, order, axis=1), np.take_along_axis(qy, order, axis=1)
    idx = np.arange(n)[None, :]
    nxt = np.where(idx + 1 < cnt[:, None], idx + 1, 0)
    cross = qx * np.take_along_axis(qy, nxt, axis=1) - np.take_along_axis(qx, nxt, axis=1) * qy
    area = 0.5 * np.abs(np.where(idx < cnt[:, None], cross, 0.0).sum(axis=1))
    return np.where(collinear, 0.0, area)


_CCW_ERRBOUND_A = (3.0 + 16.0 * 2.0**-53) * 2.0**-53   # Shewchuk (1997) orient2d static filter bound


def _exactly_collinear(p):                          # (R, n, 2) float64 -> (R,) bool, EXACT on the given floats (A2)
    """Adaptive exact orientation test (Shewchuk 1997): a vectorised float filter settles every row with a
    certainly non-zero orientation; only the rows it cannot settle are re-checked in exact rational arithmetic."""
    r_count = p.shape[0]
    rel = p - p[:, :1, :]                             # p_j - p_0 (rounded; the filter bound accounts for it)
    far = np.argmax(np.hypot(rel[..., 0], rel[..., 1]), axis=1)
    v = rel[np.arange(r_count), far]                  # p_far - p_0
    det_l = v[:, None, 0] * rel[..., 1]
    det_r = v[:, None, 1] * rel[..., 0]
    certainly_nonzero = np.abs(det_l - det_r) > _CCW_ERRBOUND_A * (np.abs(det_l) + np.abs(det_r))
    out = np.zeros(r_count, dtype=bool)
    for r in np.flatnonzero(~certainly_nonzero.any(axis=1)):   # near-collinear rows only -- rare by construction
        out[r] = _collinear_exact(p[r])
    return out


def _collinear_exact(pts):                          # (n, 2) -> bool, exact rational arithmetic on the float inputs
    q = [(Fraction(float(x)), Fraction(float(y))) for x, y in pts]
    x0, y0 = q[0]
    ref = next(((x, y) for x, y in q[1:] if (x, y) != (x0, y0)), None)
    if ref is None:
        return True                                   # all points coincide
    dx, dy = ref[0] - x0, ref[1] - y0
    return all(dx * (y - y0) - dy * (x - x0) == 0 for x, y in q)


def hull_area_batch(pos, counts):
    out = np.full(counts.shape[0], np.nan)
    for n in np.unique(counts):
        if n >= 3:
            rows = np.flatnonzero(counts == n)
            out[rows] = _hull_area_fixed_n(pos[rows, :n, :])
    return out
```

  `collective_from_positions` loops `for n in np.unique(counts)` (at most about 25 distinct counts, never per frame).
  For each `n` it takes the compact slices `xs = pos[rows, :n, 0]`, `ys = pos[rows, :n, 1]` and computes:

  | Output | Formula | Minimum n |
  |---|---|---|
  | `centroid_x`, `centroid_y` | `cx = np.mean(xs, axis=1)`, `cy = np.mean(ys, axis=1)` | 1 |
  | `team_length`, `team_width` | `max − min` per axis | 1 |
  | `stretch_index` | `np.mean(np.sqrt((xs - cx[:, None]) ** 2 + (ys - cy[:, None]) ** 2), axis=1)` | 1 |
  | `stretch_x`, `stretch_y` | `np.mean(np.abs(xs - cx[:, None]), axis=1)` and the y twin | 1 |
  | `spread` | `np.sqrt(n * np.sum((xs - cx[:, None]) ** 2 + (ys - cy[:, None]) ** 2, axis=1))` | 2 |
  | `convex_hull_area` | `_hull_area_fixed_n(pos[rows, :n, :])` | 3 |

  `back_line_batch` follows the same count-bucket pattern:

```python
def back_line_batch(pos, counts, defends_x0, *, n, adaptive_max_n):
    g = counts.shape[0]
    out = {k: np.full(g, np.nan) for k in BACK_LINE_VARIABLES[:-1]}
    back_n = np.zeros(g, dtype=np.int64)
    valid = counts >= 3
    for p in np.unique(counts[valid]):
        rows = np.flatnonzero(counts == p)
        xs, ys, d0 = pos[rows, :p, 0], pos[rows, :p, 1], defends_x0[rows]
        order = np.empty(xs.shape, dtype=np.intp)
        if d0.any():
            order[d0] = np.argsort(xs[d0], axis=1)           # DEFAULT kind == legacy np.argsort(xs)   (C1)
        if (~d0).any():
            order[~d0] = np.argsort(-xs[~d0], axis=1)        # legacy np.argsort(-xs) -- NOT a reversed ascending sort
        xs_s, ys_s = np.take_along_axis(xs, order, axis=1), np.take_along_axis(ys, order, axis=1)
        n_eff = _select_n_batch(xs_s, n, adaptive_max_n, int(p))
        for k in np.unique(n_eff):
            sub = np.flatnonzero(n_eff == k)
            sx, sy, r = xs_s[sub, :k], ys_s[sub, :k], rows[sub]
            out["defensive_line_x"][r] = np.mean(sx, axis=1)
            out["compactness_x"][r] = np.max(sx, axis=1) - np.min(sx, axis=1)
            out["back_line_high_x"][r] = np.where(d0[sub], np.max(sx, axis=1), np.min(sx, axis=1))
            out["lateral_width"][r] = np.max(sy, axis=1) - np.min(sy, axis=1)
            out["max_lateral_gap"][r] = np.max(np.diff(np.sort(sy, axis=1), axis=1), axis=1)
            back_n[r] = k
    return {**out, "back_n_count": back_n, "valid": valid}


def _select_n_batch(xs_sorted, n, adaptive_max_n, p):
    r = xs_sorted.shape[0]
    if n != "adaptive":
        return np.full(r, min(int(n), p), dtype=np.int64)
    if p in (3, 4):
        return np.full(r, p, dtype=np.int64)
    gaps = np.diff(xs_sorted, axis=1)
    cand = [c for c in (3, 4, 5) if (c - 1) < p - 1 and c <= adaptive_max_n]
    default = min(4, p)
    if not cand:
        return np.full(r, default, dtype=np.int64)
    cut = np.abs(gaps[:, [c - 1 for c in cand]])
    mx = cut.max(axis=1)
    second = (-np.sort(-cut, axis=1))[:, 1] if cut.shape[1] > 1 else np.zeros(r)
    pick = np.asarray(cand)[np.argmax(cut, axis=1)]          # first occurrence == list.index(max)
    dominant = (second == 0.0) | (mx >= 1.5 * second)
    return np.where(mx == 0.0, default, np.where(dominant, pick, default)).astype(np.int64)
```

  **Coordinate reads** by callers use `.to_numpy(dtype="float64")` (ADR-106). The kernels take arrays only. The module
  imports numpy plus the standard library's `fractions.Fraction`, which A2's exact fallback needs.
- [ ] **Step 4: Run** the kernel tests → **PASS**. Also run Task 1's parity file; it is still on the legacy
  production code, so it passes unchanged.
- [ ] **Step 5:** ruff + format + pyright on both files.

---

### Task 3: Delegate `compute_team_shape` and `compute_defensive_line`; add `compute_collective_variables`

**Files:**
- Modify:
  - `silly_kicks/tracking/_team_shape.py:76-170`
  - `silly_kicks/tracking/_defensive_line.py:213-296`
  - `silly_kicks/tracking/_collective.py` (add `compute_collective_variables`)
  - `silly_kicks/tracking/__init__.py`
- Test:
  - `tests/tracking/test_collective_parity.py` (unchanged; now the gate)
  - `tests/tracking/test_collective_variables.py`
  - `tests/tracking/test_collective_delegation.py`

**Interfaces:**
- Produces:
  - `compute_collective_variables(frames: pd.DataFrame, *, include_goalkeeper: bool = False) -> pd.DataFrame`.
    Its columns are `game_id, period_id, frame_id, team_id, n_players` (`Int64`) plus `COLLECTIVE_VARIABLES`, and it
    emits one row per `(game_id, period_id, frame_id, team_id)` with at least one valid player.
  - Exported from `silly_kicks.tracking`: `compute_collective_variables`, `collective_from_positions`,
    `COLLECTIVE_VARIABLES`.
- Consumes: Task 2's kernels.

- [ ] **Step 1: Write the new failing tests.**
  - `tests/tracking/test_collective_variables.py`:
    - `test_contract_columns_and_dtypes`.
    - `test_include_goalkeeper_changes_n_players` (both sides).
    - `test_equals_kernel_on_packed_rows`.
    - `test_ball_rows_and_nan_positions_excluded`.
    - `test_missing_required_column_raises` (a `ValueError` naming the column).
    - `test_id_dtype_invariance` (int / string / `category` team ids give equal values after `restore_id_dtype`).
    - `test_does_not_mutate_input`: the frames equal a deep copy afterwards.
  - `tests/tracking/test_collective_delegation.py`:
    - `test_add_team_shape_only_hull_area_moves`:
      - Run `add_team_shape` on the action/frame fixture already used by
        `tests/tracking/test_action_ltr_mirror_invariance.py:245` (reuse its builder).
      - Run it once with production code, and once with
        `monkeypatch.setattr("silly_kicks.tracking.features.compute_team_shape", legacy_compute_team_shape)`.
      - Assert every column except `team_shape_convex_hull_area_{attacking,defending}` is exactly equal, and those
        two are within `rtol=1e-9`.
    - `test_restdefense_output_unchanged`:
      - Run `compute_rest_defense(*make_rest_defense_fixture())` (`tests/restdefense/_fixtures.py:141`).
      - Run it with production code and with both legacy functions monkeypatched into
        `silly_kicks.restdefense._compute`.
      - Assert `assert_frame_equal(check_exact=True)` on the samples. The reports compare equal too.
    - `test_defensive_line_goal_lookups_scale_with_teams_not_frames`:
      - Wrap `GoalMap.get` with `tests/_perf_structural.call_counter(monkeypatch, GoalMap, "get")`.
      - Build synthetic frame sets of 10 and 100 frames (2 teams, 10 outfielders each).
      - Assert the lookup count is identical for both sizes and equals the number of `(game, period, team)` groups
        with at least 3 players. The legacy loop called `get` once per frame-team, so this pins the vectorisation.
- [ ] **Step 2: Run** the new files → **FAIL**: `compute_collective_variables` is missing, and the lookup count grows
  with frames.
- [ ] **Step 3: Delegate `compute_team_shape`.** Keep the signature, the empty-input returns, the `mask` filter and
  `_RESULT_COLS`. Replace the loop's geometry with the kernel and keep the Ward clustering per group:

```python
    gb = outfield.groupby(["game_id", "period_id", "frame_id"], dropna=False, sort=True, observed=True)
    codes = gb.ngroup().to_numpy()
    key_tuples = gb.size().index.tolist()                  # same order as ngroup codes == legacy iteration order
    pos, counts, first_row = pack_groups(
        codes, outfield["x"].to_numpy(dtype="float64"), outfield["y"].to_numpy(dtype="float64"), len(key_tuples)
    )
    cv = collective_from_positions(pos, counts)
    directions = (
        outfield["team_attacking_direction"].to_numpy()[first_row]
        if "team_attacking_direction" in outfield.columns else np.full(len(key_tuples), None)
    )
    # Ward line clustering stays per group (legacy semantics); it reads the packed row, never rescans.
    def_line, gap_1, gap_2 = _ward_lines(pos, counts, directions, n_defensive_lines)
    result = pd.DataFrame(key_tuples, columns=["game_id", "period_id", "frame_id"])
    # then: team_id (the argument), n_outfield_players (Int64), centroid_x, centroid_y, convex_hull_area,
    # team_length, team_width, stretch_index, defensive_line_height, inter_line_gap_1, inter_line_gap_2 -- in
    # _RESULT_COLS order
```

  `_ward_lines` is the legacy lines 127–147 moved verbatim into a helper. It loops over groups (Ward's `linkage` has
  no batch form) and reads `xs = pos[g, :counts[g], 0]`.

  The key columns are built from `key_tuples` through `pd.DataFrame(list_of_tuples)`. That reproduces the legacy
  `pd.DataFrame(rows)` dtype inference, which the parity test checks with `check_dtype=True`. If a fixture ever
  disagrees, the gate names it; never relax `check_dtype`.
- [ ] **Step 4: Delegate `compute_defensive_line`.**
  - Keep the validation block and `result_cols`. Keep the outfield filter exactly: x-valid only; y may be NaN, and
    NaN y propagates as before.
  - Group once:
    ```python
    gb = outfield.groupby(["game_id", "period_id", "frame_id", "team_id"], dropna=False, sort=True, observed=True)
    ```
  - Pack the groups.
  - Resolve ends **once per `(game, period, team)`**:
    - `ends = {k: goal_map.get(*k, allow_guess=True) for k in unique (game, period, team) of groups with count >= 3}`.
    - If any group with count ≥ 3 has `None`, raise `GoalEndUnresolvedError` with the legacy message for the **first
      such group in key order**. That is the group the legacy loop raised on.
  - `defends_x0 = end == 0.0`.
  - Call `back_line_batch`.
  - Build the output from `key_tuples`. Groups with count < 3 carry NaN metrics and `back_n_count` `<NA>`
    (`astype("Int64")`).
- [ ] **Step 5: Add `compute_collective_variables`** to `_collective.py`:
  - Required columns: `game_id, period_id, frame_id, team_id, is_ball, is_goalkeeper, x, y`. A missing column raises
    `ValueError` naming it.
  - Rows: non-ball rows with x and y valid, outfield only unless `include_goalkeeper`.
  - Grouping: `(game_id, period_id, frame_id, team_id)` with `observed=True, dropna=False, sort=True`.
  - Computation: pack, then `collective_from_positions`.
  - Output: keys from the group tuples, `n_players` `Int64`, ids restored with `id_compat.restore_id_dtype`.
  - An Examples doctest on a 2-frame, 2-team toy frame showing `centroid_x` and `spread`.
- [ ] **Step 6: Export.** Add `compute_collective_variables`, `collective_from_positions` and `COLLECTIVE_VARIABLES`
  to `silly_kicks/tracking/__init__.py` (import block and `__all__`, alphabetical).
- [ ] **Step 7: Run.**
  ```
  .venv/Scripts/python -m pytest tests/tracking/test_collective_parity.py tests/tracking/test_collective_kernels.py tests/tracking/test_collective_variables.py tests/tracking/test_collective_delegation.py tests/test_c4_aggregator_count.py tests/test_public_api_examples.py -q -p no:randomly
  ```
  → **PASS**. The aggregator count stays 33: no `add_*` is added.

  Then run every caller's suite (§7.16):
  ```
  .venv/Scripts/python -m pytest tests/tracking tests/restdefense tests/causal -m "not e2e" -q -p no:randomly --benchmark-skip
  ```
  → PASS.
- [ ] **Step 8: Re-measure** Task 0 Step 4's timings. Record before/after in the ADR draft (Task 24).
- [ ] **Step 9:** ruff + format + pyright on every touched file.

---

### Task 4: `tracking.preprocess` — zero-phase Butterworth, resampling, residual analysis

**Files:**
- Create: `silly_kicks/tracking/preprocess/_butterworth.py`
- Modify:
  - `silly_kicks/tracking/preprocess/_config_dataclass.py:17,43-70` (`SmoothingMethod`, two fields, `__post_init__`)
  - `silly_kicks/tracking/preprocess/_smoothing.py:22-26,63-127` (tag and dispatch)
  - `silly_kicks/tracking/preprocess/__init__.py`
  - `silly_kicks/tracking/__init__.py`
- Test: `tests/tracking/test_preprocess_butterworth.py`

**Interfaces — produces (consumed by Tasks 16 and 20):**

```python
def winter_correction(order: int) -> float                       # (2**0.5 - 1) ** (1 / (2 * order))  (Winter 2009)
def butterworth_min_length(fs: float, cutoff_hz: float, order: int = 3) -> int   # sosfiltfilt default padlen + 1
def butterworth_lowpass(values: np.ndarray, fs: float, cutoff_hz: float, order: int = 3) -> np.ndarray
def resample_uniform(t: np.ndarray, values: np.ndarray, fs_out: float, run_bounds: np.ndarray, *,
                     n_out: int | None = None) -> np.ndarray      # grid k / fs_out, k = 0..n_out-1; NaN outside runs
def residual_analysis_cutoff(values: np.ndarray, fs: float, grid: np.ndarray, *, order: int = 3,
                             tail_fraction: float = 0.5) -> float
def resample_frames(frames: pd.DataFrame, target_hz: float, *, max_gap_seconds: float = 0.5) -> pd.DataFrame
# PreprocessConfig gains: butterworth_cutoff_hz: float = 0.4, butterworth_order: int = 3
# SmoothingMethod = Literal["savgol", "ema", "butterworth", None]
```

C18 (concretisation) has two parts, both documented in the docstrings:
- **Evaluable grid.** `residual_analysis_cutoff` evaluates only grid frequencies whose design frequency
  (`f / winter_correction(order)`) is below the Nyquist frequency of `fs`. At 10 Hz the spec's 0.1–5.0 Hz grid
  therefore ends near 4.29 Hz.
- **Noise tail.** The "linear high-frequency tail" is the upper half of the evaluated grid (`tail_fraction = 0.5`).

- [ ] **Step 1: Write the failing tests** (§9.1 Butterworth bullets):
  - `test_winter_correction_value`: `winter_correction(3) == (2**0.5-1)**(1/6)`.
  - `test_zero_phase_lag_on_in_band_sinusoid`:
    - Input: a 0.05 Hz sinusoid at 10 Hz, 600 s, cutoff 0.4.
    - Assert: the cross-correlation peak of input vs output is at lag 0, and the interior amplitude ratio is
      > 0.999.
  - `test_dual_pass_minus_3db_at_cutoff`:
    - Frequency response: the designed sos's `|H(f_c)|²` from `scipy.signal.sosfreqz` is the dual-pass gain, and it
      is within 1% of `1/√2`.
    - Empirical: the steady-state amplitude of a long sinusoid at the cutoff after `butterworth_lowpass` is within 2%
      of `1/√2`.
  - `test_cutoff_at_or_above_nyquist_raises`: at `fs = 10`, cutoff 5.0 raises. So does cutoff 4.5, whose design
    frequency 4.5/0.858 = 5.24 is above Nyquist. Both raise `ValueError` naming the Nyquist frequency.
  - `test_short_series_raises_with_min_length`: `len == butterworth_min_length(...) - 1` raises; `len == min_length`
    passes. Both sides of the band.
  - `test_residual_analysis_recovers_planted_cutoff`:
    - Input: a signal band-limited at 0.6 Hz plus white noise (σ 0.05) at 25 Hz.
    - Grid: 0.1–5.0 in 0.05 steps.
    - Assert: the recovered cutoff is within ±0.15 Hz of 0.6.
  - `test_residual_analysis_noise_free_input` pins the documented behaviour with σ = 0: `ValueError` "never reaches
    the noise-line intercept", or the grid minimum. The docstring states which one, and the test asserts it.
  - `test_residual_analysis_skips_grid_points_above_nyquist` (C18): at 10 Hz no raise, and the evaluated grid ends
    below `5.0 * winter_correction(3)`.
  - `test_resample_uniform_exact_on_linear`: a linear signal is reproduced within `rtol=1e-12` on the grid inside
    runs.
  - `test_resample_never_crosses_split`: two runs with a gap. The grid points inside the gap are NaN, and grid points
    near each run end use only that run's samples.
  - `test_smooth_frames_butterworth_additive_and_tagged`:
    - Output: `x`/`y` are bit-identical; `x_smoothed`/`y_smoothed` are present; `_preprocessed_with ==
      "method=butterworth|bw_cutoff_hz=0.4|bw_order=3"`.
    - A re-call is idempotent.
    - The savgol and ema tags are unchanged: assert their exact existing strings.
  - `test_smooth_frames_short_group_passes_through`.
  - `test_preprocess_config_butterworth_fields`: defaults 0.4/3; `butterworth_cutoff_hz <= 0` raises; `order < 1`
    raises; `derive_velocity=True` with `smoothing_method="butterworth"` is accepted.
  - `test_resample_frames_contract`:
    - `frame_id == k`, `time_seconds == k / target_hz`, `frame_rate == target_hz`.
    - `x`/`y` are interpolated within runs split at gaps > `max_gap_seconds`.
    - Every other column is step-held from the latest native row at or before the grid time.
    - `speed`, `vx` and `vy`, where present, are NaN (re-derive after resampling).
    - One row per grid point per `(period, player or ball)` inside a run.
- [ ] **Step 2: Run** → **FAIL** (import errors).
- [ ] **Step 3: Implement.** `butterworth_lowpass` designs `butter(order, cutoff_hz / winter_correction(order),
  btype="low", fs=fs, output="sos")`, then applies `sosfiltfilt(sos, values)`. It raises when the design frequency is
  at or above `fs / 2`, and when `len(values) < butterworth_min_length(...)`. `butterworth_min_length` mirrors
  `sosfiltfilt`'s default pad:

```python
def butterworth_min_length(fs, cutoff_hz, order=3):
    sos = _design(fs, cutoff_hz, order)
    ntaps = 2 * sos.shape[0] + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    return 3 * ntaps + 1                      # sosfiltfilt requires len > padlen = 3 * ntaps
```

  `resample_uniform` finds each run's native slice with `np.searchsorted` (no rescan) and calls
  `np.interp(grid[g], t[lo:hi], v[lo:hi])` for grid points inside `[start, end]`. `residual_analysis_cutoff`:

```python
def residual_analysis_cutoff(values, fs, grid, *, order=3, tail_fraction=0.5):
    nyq = fs / 2.0
    grid = np.sort(np.asarray(grid, dtype=np.float64))
    grid = grid[grid / winter_correction(order) < nyq]                    # C18: the evaluable grid at this rate
    resid = np.array([np.sqrt(np.mean((values - butterworth_lowpass(values, fs, f, order)) ** 2)) for f in grid])
    tail = grid >= grid.max() * tail_fraction                              # C18: "linear high-frequency tail"
    _slope, intercept = np.polyfit(grid[tail], resid[tail], 1)             # intercept = noise RMS (Winter)
    below = np.flatnonzero(resid <= intercept)
    if below.size == 0:
        raise ValueError("residual analysis: R(f) never reaches the noise-line intercept on this grid")
    i = int(below[0])
    if i == 0:
        return float(grid[0])
    f0, f1, r0, r1 = grid[i - 1], grid[i], resid[i - 1], resid[i]
    return float(f0 + (r0 - intercept) * (f1 - f0) / (r0 - r1))            # linear interpolation to the crossing
```

  `smooth_frames`:
  - `method_used == "butterworth"` calls `_butterworth_per_group(values, hz, cfg)` inside the existing group loop.
    It mirrors `_savgol_per_group`'s NaN handling: interior NaN is linearly bridged, filtered, then restored. A group
    shorter than `butterworth_min_length` passes through unchanged.
  - `_provenance_tag` returns the new `method=butterworth|bw_cutoff_hz=…|bw_order=…` string **only** for
    butterworth, so the savgol/ema tags stay byte-identical (Hyrum's law).
- [ ] **Step 4: Export** the four new public functions (`butterworth_lowpass`, `resample_uniform`,
  `residual_analysis_cutoff`, `resample_frames`) from `tracking/preprocess/__init__.py` and `tracking/__init__.py`
  (alphabetical `__all__`). `winter_correction` and `butterworth_min_length` stay module-level in the private
  `_butterworth.py`, and Task 16 imports them from there.
- [ ] **Step 5: Regenerator check.** Run `.venv/Scripts/python scripts/regenerate_provider_defaults.py`, then
  `git diff --exit-code silly_kicks/tracking/preprocess/_provider_defaults_generated.py` → no diff (the new fields
  default). Run `tests/test_preprocess_baseline_integrity.py` and every existing `smooth_frames`/`derive_velocities`
  test.
- [ ] **Step 6: Run** the new test file plus every `git grep -l "smooth_frames\|derive_velocities" tests` hit →
  **PASS**. Then ruff + format + pyright.

---

### Task 5: Dead-ball taxonomy (D20) and array goal-relative transforms (C2)

**Files:**
- Modify:
  - `silly_kicks/tracking/_provider_visibility.py` (after l.26)
  - `silly_kicks/tracking/_geometry.py` (after l.111)
- Test:
  - `tests/tracking/test_dead_ball_taxonomy.py`
  - `tests/tracking/test_geometry_array_twins.py`

**Interfaces — produces:**

```python
_DEAD_BALL_OBSERVED_PROVIDERS = frozenset({"sportec", "idsse", "gradientsports"})
def dead_ball_observed(provider: str) -> bool          # validate_provider(provider) first: unclassified -> ValueError
def to_goal_relative_x_array(x: np.ndarray, *, goal_x: float) -> np.ndarray   # (105 - x) if _flip(goal_x) else x (copy)
def to_goal_relative_y_array(y: np.ndarray, *, goal_x: float) -> np.ndarray   # (68 - y)  if _flip(goal_x) else y (copy)
```

- [ ] **Step 1: Write the failing tests.**
  - Taxonomy:
    - `test_observed_members`.
    - `test_skillcorner_and_metrica_unobserved`.
    - `test_unclassified_provider_raises` (a `ValueError` naming both taxonomies).
    - `test_observed_is_subset_of_classified` (every member passes `validate_provider`).
    - `test_snapshot_is_not_classified`: `"snapshot"` raises. Freeze frames are refused earlier (§7.3), never
      treated as observed.
  - Geometry:
    - `test_array_twins_equal_scalar_elementwise`: random x/y including NaN, both `goal_x ∈ {0.0, 105.0}`, compared
      with `np.testing.assert_array_equal` against `[to_goal_relative_x(v, goal_x=g) for v in x]`.
    - `test_array_twins_are_point_reflection`: applying the transform twice returns the input.
    - `test_array_twins_do_not_mutate_input`.
    - `test_geometry_version_unchanged`: it is still `"goal-relative-2"`.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement.** Both transforms reuse `_flip` and `FIELD_LENGTH`/`PITCH_WIDTH`. Each docstring carries
  an Examples doctest.
- [ ] **Step 4: Run** → **PASS**. Also run `tests/tracking/test_detection_aware_visibility.py` and
  `tests/scripts/test_trainer_cache_and_providers.py` (the existing `_provider_visibility` consumers). Then ruff +
  format + pyright.

---

### Task 6: Re-key the metric-contract completeness gate (ADR-098 amendment)

**Files:**
- Modify: `tests/test_metric_contracts.py:22-50, 86-93, 142-152`

- [ ] **Step 1: Write the failing test first.** Add
  `test_completeness_is_keyed_per_exported_constant_planted(tmp_path)`:
  - It builds a fake package tree in `tmp_path`. One package's `__init__.py` has
    `__all__ = ["A_KEYS", "A_METRIC_COLUMNS", "B_METRIC_COLUMNS"]`.
  - It asserts `_metric_constants_exported(tmp_path) == {("pkg", "A_METRIC_COLUMNS"), ("pkg", "B_METRIC_COLUMNS")}`.
  - It asserts that `_completeness_diff(registered={("pkg", "A_METRIC_COLUMNS")}, derived=...)` is non-empty.
- [ ] **Step 2: Run** → **FAIL** (`_metric_constants_exported` is undefined).
- [ ] **Step 3: Implement.**
  - `_metric_constants_exported(root: pathlib.Path) -> set[tuple[str, str]]` walks `root.iterdir()` exactly as
    `_packages_exporting_metric_columns` does, but yields `(package, constant)` for **every** `*_METRIC_COLUMNS` in
    `__all__`.
  - `_completeness_diff(registered, derived)` returns `registered ^ derived`.
  - `test_completeness_three_bucket` asserts:

    ```python
    derived = _metric_constants_exported(pathlib.Path(silly_kicks.__file__).parent)
    registered = {(module.rsplit(".", 1)[-1], metric_attr) for module, _k, metric_attr, _f in _PKG.values()}
    assert not _completeness_diff(registered, derived), _completeness_diff(registered, derived)
    assert len(registered) == len(_PKG)                    # one family per exported constant
    ```

  - The `_EXEMPT` / `_UNDERIVABLE` buckets are kept unchanged.
  - The old package-level helper is deleted, since it has no other caller (`git grep`). Every existing family still
    maps one-to-one; measured at plan time, each of the 8 packages exports exactly one `*_METRIC_COLUMNS`.
- [ ] **Step 4: Run** `tests/test_metric_contracts.py` → **PASS**. Then ruff + format + pyright.

---

### Task 7: Extract shared driver helpers (`scripts/_reliability.py`, `scripts/_corpus_visibility.py`)

**Files:**
- Create: `scripts/_reliability.py`, `scripts/_corpus_visibility.py`
- Modify:
  - `scripts/validate_team_kpi_reliability.py:64-115,169-197`
  - `scripts/validate_gk_decision.py:44-63`
  - `scripts/train_ghost_gk.py` (the `validate_corpus_visibility` definition)
- Test: `tests/scripts/test_reliability_module.py`, `tests/scripts/test_corpus_visibility_module.py`

- [ ] **Step 1: Write the failing tests.**
  - `test_reliability_functions_are_shared`:
    - Import the three modules with the idiom of `tests/scripts/test_team_kpi_reliability.py:14-19`.
    - Assert `v.icc1 is r.icc1`, `g.icc1 is r.icc1`, `v.split_half_reliability is r.split_half_reliability`,
      `v.type_ii_slope is r.type_ii_slope`, `v.compare_providers is r.compare_providers`.
  - `test_reliability_values_unchanged`: seeded inputs give the same values as literal expectations captured
    **before** the move. Capture them in Step 0 below.
  - `test_train_ghost_gk_reimports_validate_corpus_visibility`:
    - Load `scripts/train_ghost_gk.py` by path, using the idiom of
      `tests/scripts/test_trainer_cache_and_providers.py:180-190`.
    - Assert `t.validate_corpus_visibility is _corpus_visibility.validate_corpus_visibility`.
    - The re-import is load-bearing (§7.1 item 5).
- [ ] **Step 0 (before Step 1's run): Capture expectations.** Run the seeded inputs through the **unmoved** functions
  in a scratch session. Paste the literal outputs into `test_reliability_values_unchanged`. This proves "identical
  code" by value, not only by object identity.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Move the code byte-for-byte.**
  - `icc1`, `_stable_half`, `split_half_reliability`, `type_ii_slope` and `compare_providers` go to
    `_reliability.py`. `split_half_reliability` needs `_stable_half`.
  - Before deleting `validate_gk_decision.icc1`, prove it identical: extract both function bodies to temp files and
    run `git diff --no-index` → no output. If they differ, stop and report to the owner (§7.1 item 4 asserts they
    are identical).
  - The callers import the names from `scripts._reliability` unchanged.
  - `validate_corpus_visibility` moves to `_corpus_visibility.py`; `train_ghost_gk.py` imports it under the same
    name.
- [ ] **Step 4: Run.**
  ```
  .venv/Scripts/python -m pytest tests/scripts/test_reliability_module.py tests/scripts/test_corpus_visibility_module.py tests/scripts/test_team_kpi_reliability.py tests/scripts/test_gk_decision_battery_kernels.py tests/scripts/test_trainer_cache_and_providers.py tests/scripts/test_corpus_driver_resilience.py tests/scripts/test_provenance_wiring.py -q
  ```
  → **PASS**. `test_corpus_driver_resilience.py` runs load rules A–D, which scan private scripts modules too. Then
  ruff + format + pyright.

---

## Coordination kernels (Tasks 8–13)

Every kernel lives in `silly_kicks/coordination/_kernels/`.
- **Imports:** only `numpy`, `scipy`, the standard library and (in `_numba.py`) optionally `numba` (§7.1).
- **Tests:** in `tests/coordination/kernels/`. They use no mocks and no pandas.
- **Package scaffolding:** Task 8 creates `silly_kicks/coordination/__init__.py` with a module docstring and
  `__all__ = []` (filled in Task 18), plus the empty `silly_kicks/coordination/_kernels/__init__.py`,
  `tests/coordination/__init__.py` and `tests/coordination/kernels/__init__.py`.

### Task 8: Circular statistics and phase (`_circular.py`, `_phase.py`)

**Files:**
- Create:
  - `silly_kicks/coordination/__init__.py`
  - `silly_kicks/coordination/_kernels/__init__.py`
  - `silly_kicks/coordination/_kernels/_circular.py`
  - `silly_kicks/coordination/_kernels/_phase.py`
  - `tests/coordination/__init__.py`
  - `tests/coordination/kernels/__init__.py`
- Test: `tests/coordination/kernels/test_circular.py`, `tests/coordination/kernels/test_phase.py`

**Interfaces — produces:**

```python
# _circular.py
HIST_BIN_CENTRES_DEG: tuple[int, ...] = (-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150)
HIST_BIN_LABELS: tuple[str, ...] = ("m180", "m150", "m120", "m090", "m060", "m030",
                                    "p000", "p030", "p060", "p090", "p120", "p150")
def wrap_deg(a: np.ndarray) -> np.ndarray                          # to (-180, 180]
def circular_summary(z_sum: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # (mean_deg = deg(arg(z_sum)), R = |z_sum| / n, circ_sd_deg = deg(sqrt(-2 ln R))); n == 0 -> NaN; R == 0 -> +inf
def hist_bin_index(z: np.ndarray) -> np.ndarray                  # int8 in 0..11 (index into HIST_BIN_CENTRES_DEG)
def near_in_phase(z: np.ndarray, near_deg: float) -> np.ndarray   # bool: Re(z) >= cos(radians(near_deg)); |z| == 1
# _phase.py
def pad_length(n: int, fs: float, band_low_cpm: float) -> int     # min(n, round(60 * fs / band_low_cpm))
def analytic_phase(x: np.ndarray, pad: int) -> np.ndarray         # centre; np.pad(mode="reflect"); scipy.signal.hilbert;
                                                                  # strip pad; angle in (-pi, pi]
def phasor(theta: np.ndarray) -> np.ndarray                       # exp(1j * theta)
def phase_valid_fraction(theta: np.ndarray) -> float              # mean(diff(unwrap(theta)) > 0); NaN if len < 2
```

`hist_bin_index` implements Bourbousson's bins with the ±180° wrap in one expression:

```python
def hist_bin_index(z):
    a = np.degrees(np.angle(z))                                    # (-180, 180]
    return (np.floor((a + 195.0) / 30.0).astype(np.int64) % 12).astype(np.int8)
    # a in [-180, -165) -> 0 ; [-165, -135) -> 1 ; ... ; [165, 180] -> 12 % 12 = 0  (the -180 bin wraps)
```

- [ ] **Step 1: Write the failing tests.**
  - `test_circular_summary_parity_with_scipy`: random angles; `mean_deg` matches `scipy.stats.circmean`, and
    `circ_sd_deg` matches `scipy.stats.circstd` (both converted to degrees, `atol=1e-10`).
  - `test_histogram_edges_both_sides`:
    - For each centre `c`, `c−15+1e-9` → that bin and `c−15−1e-9` → the previous bin.
    - The wrap: `165+1e-9` → bin 0 and `165−1e-9` → bin 11; `-180+1e-9` → bin 0; `180` → bin 0.
  - `test_near_in_phase_threshold_both_sides`: `29.999°` → True, `30.001°` → False, `-29.999°` → True,
    `-30.001°` → False.
  - `test_known_offset_sinusoids` (§9.1):
    - Input: A = sin(2π·0.02·t), B = sin(2π·0.02·t − 40°), at 10 Hz for 3,000 s, pad by `pad_length(n, 10, 0.22)`.
      (3,000 s, not the §9.1 example's 1,200 s: the reflect pad slightly perturbs an otherwise FFT-exact clean
      sinusoid, so R clears 0.999 only with more samples. TF-58 Task 8, owner-approved.)
    - The relative phase φ = arg(z_A · conj z_B) has circular mean within 0.5° of +40° and R > 0.999.
  - `test_anti_phase_is_180`: the mean is within 0.5° of ±180°.
  - `test_noise_lowers_R_monotonically`: σ ∈ (0, 0.2, 0.5, 1.0), seeded, averaged over 20 draws, gives strictly
    decreasing R.
  - `test_mean_centring_is_load_bearing` (non-vacuity of the pre-Hilbert step): on a DC-offset sinusoid the kernel
    (which centres) tracks the true phase, while an un-centred Hilbert is badly biased (central-window RMS > 10×).
    This is the robust, load-bearing benefit — not the reflect padding.
  - `test_reflect_padding_reduces_edge_error` (non-vacuity, representative regime): on a ~1-minute-rhythm slow window
    (0.015 Hz, 90 s = 1.35 cycles — the realistic non-integer-cycle case), the padding branch changes the edge phase
    and reduces the first/last-10% RMS vs `pad=0` (measured ~2.1×; asserted > 1.5×). The edge benefit is
    regime-dependent, not universal — integer-cycle sinusoids are artificially FFT-periodic. (TF-58 Task 8,
    owner-approved; the original flat "by at least 2×" was not a robust property.)
  - `test_pad_length_rule`: `pad_length(10_000, 10.0, 0.22) == round(60*10/0.22)`; `pad_length(100, 10.0, 0.22) == 100`.
  - `test_centring_makes_offset_irrelevant`: `analytic_phase(x + 50.0, pad) == analytic_phase(x, pad)`
    (`atol=1e-9`).
  - `test_phase_valid_fraction`: a pure sinusoid gives `> 0.99`; a seeded random walk gives `< 0.9`.
- [ ] **Step 2: Run** → **FAIL** (the modules are missing).
- [ ] **Step 3: Implement** to the interface above. In `analytic_phase`, reflect-padding is by
  `min(pad, len(x) - 1)`: numpy's `reflect` needs at least 2 samples, and shorter runs never reach here (Task 16 drops
  runs shorter than the filter minimum).
- [ ] **Step 4: Run** → **PASS**. Then ruff + format + pyright.

---

### Task 9: Lagged cross-correlation and Fisher-z pooling (`_xcorr.py`)

**Files:**
- Create: `silly_kicks/coordination/_kernels/_xcorr.py`
- Test: `tests/coordination/kernels/test_xcorr.py`

**Interfaces — produces:**

```python
def lagged_pearson(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]
    # r (2L+1,) float64 for lags -L..L, r[l+L] = corr(a[t], b[t+l]) over the overlap; n (2L+1,) int64 overlap sizes
def fisher_pool(r: np.ndarray, n: np.ndarray) -> np.ndarray      # (S, 2L+1) x2 -> (2L+1,); weights n - 3 (<= 0 dropped)
def xcorr_summary(r: np.ndarray, fs: float) -> tuple[float, float, float, float]
    # (max_abs_r, lag_s, r_at_max, r_lag0); ties -> smallest |lag|, then the negative lag; all-NaN -> 4 x NaN
def min_slice_samples(max_lag: int) -> int                        # 4 * max_lag  (§7.8.2)
```

Implementation — exact overlap Pearson from FFT sums and prefix sums:
- Centre `a` and `b` by their global means first. Pearson is shift-invariant, and centring removes the cancellation
  that raw positions (~50 m) would cause.
- `Sab(ℓ) = scipy.signal.correlate(b, a, mode="full", method="fft")[ℓ + N − 1]`, which equals
  `Σ_t a[t]·b[t+ℓ]`.
- `Sa`, `Sb`, `Saa`, `Sbb` over each lag's overlap come from `np.cumsum` prefix sums:
  - for `ℓ ≥ 0`: `a[0:N−ℓ]`, `b[ℓ:N]`;
  - for `ℓ < 0`: `a[−ℓ:N]`, `b[0:N+ℓ]`.
- `r = (n·Sab − Sa·Sb) / sqrt((n·Saa − Sa²)(n·Sbb − Sb²))`.
- A zero-variance side gives NaN. The compute layer maps it to `degenerate_constant`.

```python
def fisher_pool(r, n):
    w = np.where(np.isfinite(r) & (n > 3), n - 3.0, 0.0)
    z = np.arctanh(np.clip(np.where(w > 0, r, 0.0), -1.0 + 1e-15, 1.0 - 1e-15))
    sw = w.sum(axis=0)
    return np.where(sw > 0, np.tanh((w * z).sum(axis=0) / np.where(sw > 0, sw, 1.0)), np.nan)
```

- [ ] **Step 1: Write the failing tests** (§9.1):
  - `test_matches_numpy_corrcoef_per_lag`: random a, b with N = 900 and L = 150. For every ℓ, `r` equals
    `np.corrcoef` of the overlap within `atol=1e-9`, and `n` equals the overlap size.
  - `test_shifted_copy_positive_lag_when_a_leads`: `b[t] = a[t−k]` (B lags A by k samples) → `lag_s == +k/fs`,
    `r_at_max` within `1e-12` of 1, and `max_abs_r` within `1e-12` of 1.
  - `test_inverted_b_negative_r`: `b = −a` gives `r_at_max ≈ −1` and `lag_s == 0`.
  - `test_fisher_single_slice_identity`: `fisher_pool(r[None], n[None])` equals `r` within `1e-12` wherever `n > 3`.
  - `test_fisher_weights_by_n_minus_3`: a hand-computed two-slice case.
  - `test_tie_break_smallest_abs_lag_then_negative`: a constructed `r` with equal |r| at ℓ = −3, +3 and +5 → −3.
  - `test_min_slice_samples_is_four_L`.
  - `test_constant_side_is_nan`.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → **PASS**. Then ruff + format + pyright.

---

### Task 10: Vector coding (`_vector_coding.py`)

**Files:**
- Create: `silly_kicks/coordination/_kernels/_vector_coding.py`
- Test: `tests/coordination/kernels/test_vector_coding.py`

**Interfaces — produces:**

```python
PATTERNS: tuple[str, ...] = ("in_phase", "anti_phase", "a_phase", "b_phase")
def coupling_angle_deg(da: np.ndarray, db: np.ndarray) -> np.ndarray   # degrees(atan2(db, da)) % 360.0 -> [0, 360]
def classify(angle_deg: np.ndarray) -> np.ndarray                      # int8 index into PATTERNS (Moura 2016 Table 1)
def stationary_mask(da, db, eps_a: float, eps_b: float) -> np.ndarray  # (|da| < eps_a) & (|db| < eps_b)
```

The half-open Table 1 bins are exactly the eight 45° octants centred on the axes and diagonals:

```python
_OCTANT_TO_PATTERN = np.array([2, 0, 3, 1, 2, 0, 3, 1], dtype=np.int8)   # a, in, b, anti, a, in, b, anti

def classify(angle_deg):
    octant = np.floor((angle_deg + 22.5) / 45.0).astype(np.int64) % 8   # 337.5 and 360 -> octant 0 (A-phase)
    return _OCTANT_TO_PATTERN[octant]
```

Every boundary (22.5 + 45k) is exactly representable in binary, so the edges are exact.

**C19 (a correction to a spec example).** §9.1 says the printed Eq. 2 "misclassifies a 225° coupling". The printed
form gives `arctan|Δθ₂/Δθ₁| = 45°` for a 225° coupling. That is the wrong *angle*, but 45° and 225° are both in-phase
in Table 1, so the *class* is coincidentally right. The printed form misclassifies at 135° and 315°: anti-phase read
as in-phase. The regression pin asserts all three, which keeps the spec's intent (pin that the printed form is wrong)
and makes the test true.

- [ ] **Step 1: Write the failing tests.**
  - `test_table1_edges_both_sides`: for every edge `e ∈ {22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5}`,
    `classify(e − 1e-9)` and `classify(e + 1e-9)` equal the Table 1 classes, and `classify(e)` is the upper bin.
    Also `classify(0.0)` and `classify(360.0)` are `a_phase`.
  - `test_four_quadrant_angle`: the (Δa, Δb) pairs (1,1), (−1,1), (−1,−1), (1,−1) give 45, 135, 225 and 315.
  - `test_printed_eq2_is_wrong` (C19 regression pin). With `printed = degrees(arctan(abs(db/da)))`:
    - at 225°, `printed == 45 != 225` (the angle is wrong);
    - at 135° and 315°, `classify(printed) == in_phase` while `classify(true) == anti_phase` (misclassified).
  - `test_stationary_with_epsilon_both_sides`: `|da| = eps − 1e-12` with `|db|` below its epsilon → stationary;
    `eps + 1e-12` → not stationary; exactly one side below → not stationary (§7.8.3 "and").
  - `test_negative_zero_angle_maps_to_a_phase`: `atan2(-0.0, 1.0)` → A-phase, not a crash at 360.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → **PASS**. Then ruff + format + pyright.

The rule that differences never span a segment split, and the 3-sample minimum per subdivision, are compute-layer
rules; Task 17 tests them.

---

### Task 11: Spectral median frequency and pooled Welch coherence (`_spectral.py`)

**Files:**
- Create: `silly_kicks/coordination/_kernels/_spectral.py`
- Test: `tests/coordination/kernels/test_spectral.py`

**Interfaces — produces:**

```python
@dataclass(frozen=True)
class WelchSpectra:
    f: np.ndarray        # Hz
    pxx: np.ndarray
    pyy: np.ndarray
    pxy: np.ndarray      # complex cross-spectrum
    k: int               # Welch segments averaged

def median_frequency_cpm(x: np.ndarray, fs: float) -> float
def pooled_median_frequency(values: np.ndarray, durations_s: np.ndarray) -> float   # duration-weighted mean
def min_spectral_samples(fs: float, band_low_cpm: float) -> int                       # ceil(2 * 60 * fs / band_low_cpm)
def welch_segment_count(n: int, nperseg: int) -> int                                  # 0 if n < nperseg
def welch_spectra(a: np.ndarray, b: np.ndarray, fs: float, nperseg: int) -> WelchSpectra
def pooled_coherence(spectra: Sequence[WelchSpectra], band_low_cpm: float, band_high_cpm: float
                     ) -> tuple[float, float, int]                                     # (band_mean, peak_cpm, K)
```

```python
def median_frequency_cpm(x, fs):
    f, p = scipy.signal.periodogram(x, fs=fs, detrend="constant", window="boxcar")   # mean removed (§3.1 item 2)
    f, p = f[1:], p[1:]                                                                 # DC excluded
    c = np.cumsum(p)
    if c[-1] <= 0.0:
        return float("nan")                                                             # -> degenerate_constant
    half = 0.5 * c[-1]
    k = int(np.searchsorted(c, half))                                                   # first bin reaching half
    if k == 0:
        return float(60.0 * f[0])
    f_med = f[k - 1] + (half - c[k - 1]) * (f[k] - f[k - 1]) / (c[k] - c[k - 1])        # linear between bins
    return float(60.0 * f_med)


def pooled_coherence(spectra, band_low_cpm, band_high_cpm):
    k = sum(s.k for s in spectra)
    pxx = sum(s.k * s.pxx for s in spectra) / k       # pool the SPECTRA (weighted by Welch segment count) ...
    pyy = sum(s.k * s.pyy for s in spectra) / k
    pxy = sum(s.k * s.pxy for s in spectra) / k
    coh = np.abs(pxy) ** 2 / (pxx * pyy)              # ... then form the coherence -- never average coherences
    f_cpm = 60.0 * spectra[0].f
    band = (f_cpm >= band_low_cpm) & (f_cpm <= band_high_cpm)
    return float(coh[band].mean()), float(f_cpm[band][np.argmax(coh[band])]), k
```

`welch_spectra` uses `scipy.signal.welch` and `scipy.signal.csd` with `window="hann"`, `nperseg`,
`noverlap=nperseg // 2` and `detrend="constant"`. Its `k` is `welch_segment_count(len(a), nperseg)`, which equals
`1 + (n − nperseg) // (nperseg // 2)` for `n ≥ nperseg`.

- [ ] **Step 1: Write the failing tests** (§9.1):
  - `test_pure_tone_median_frequency`: a tone at f₀ = 0.5 cycles·min⁻¹, 10 Hz, 3,600 s. The result is within one
    bin (`60/3600 = 0.0167` cycles·min⁻¹) of 0.5: the linear interpolation between bins moves a pure tone by at most
    one bin.
  - `test_positive_offset_does_not_move_median`: `x + 30.0` gives the same value (mean removal).
  - `test_min_spectral_samples_is_two_periods`: `min_spectral_samples(10.0, 0.22) == ceil(2*60*10/0.22)`.
  - `test_pooled_median_is_duration_weighted`.
  - `test_constant_series_is_nan`.
  - `test_coherence_near_one_for_linearly_filtered_pair`: b is a 3-tap moving average of a; the band mean is
    > 0.95 with K ≥ 8.
  - `test_coherence_near_inverse_k_for_independent_noise`: seeded, averaged over 50 draws; the mean band coherence
    lies in `[0.5/K, 2/K]`.
  - `test_pools_spectra_not_coherences` (non-vacuity): two slices with very different SNR. The pooled-spectra result
    differs from the K-weighted mean of the per-slice coherences by > 0.05.
  - `test_segment_count_rule`.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → **PASS**. Then ruff + format + pyright.

---

### Task 12: Cluster phase and sample entropy (`_cluster.py`, `_entropy.py`, `_numba.py` part 1)

**Files:**
- Create:
  - `silly_kicks/coordination/_kernels/_cluster.py`
  - `silly_kicks/coordination/_kernels/_entropy.py`
  - `silly_kicks/coordination/_kernels/_numba.py`
- Modify: `.github/workflows/ci.yml:91` **and** the `slow` job's cache step — the numba cache key.
- Test:
  - `tests/coordination/kernels/test_cluster.py`
  - `tests/coordination/kernels/test_entropy.py`
  - `tests/coordination/kernels/test_numba_parity.py`

**Interfaces — produces:**

```python
# _cluster.py (Richardson et al. 2012)
def cluster_phase(z: np.ndarray, valid: np.ndarray, min_players: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # z (T, K) unit phasors, valid (T, K) bool ->
    #   q (T,) group phasor e^{i q(t)} (NaN+NaN*1j where unusable),
    #   rel (T, K) = z * conj(q) (0 where not valid or not usable),
    #   usable (T,) = valid.sum(axis=1) >= min_players
@dataclass(frozen=True)
class ClusterWindowStats:
    phi_bar: np.ndarray      # (K,) radians; NaN for a player with no usable sample
    rho_k: np.ndarray        # (K,)
    rho_group_i: np.ndarray  # (n_usable,) instantaneous group synchrony over the window's usable samples
    rho_group_mean: float
    rho_group_sd: float      # np.std(ddof=0): descriptive dispersion of the window's series
    n_players_mean: float
def window_cluster_stats(rel: np.ndarray, valid: np.ndarray, usable: np.ndarray, start: int, end: int
                         ) -> ClusterWindowStats

# _entropy.py (Richman & Moorman 2000)
def sampen(x: np.ndarray, m: int = 1, r_sd: float = 0.2) -> tuple[float, int, int]
    # (SampEn, A, B); r = r_sd * np.std(x, ddof=0); templates over the first N - m points; self-matches excluded;
    # A == 0 or B == 0 -> (nan, A, B)
def cross_sampen(u: np.ndarray, v: np.ndarray, m: int = 1, r: float = 0.2) -> tuple[float, int, int]
    # both z-scored (ddof=0); pairs (i from u, j from v); direction-independent

# _numba.py
HAVE_NUMBA: bool
def use_numba() -> bool                        # HAVE_NUMBA and os.environ.get("SILLY_KICKS_COORDINATION_FORCE_NUMPY") != "1"
def dominance_count_self(a: np.ndarray, b: np.ndarray, r: float) -> int          # njit Fenwick sweep
def dominance_count_cross(a1, b1, a2, b2, r: float) -> int                       # njit Fenwick sweep
```

Exact counting. The distance predicate is always the naive one, `abs(p − q) <= r` per coordinate.
- **B for m = 1 (both paths).** Sort the values, then vectorised **predicate bisection**: `s[q] − s[p]` is
  non-decreasing in `q` because rounded subtraction is monotone. A `searchsorted(s, s + r)` would test a different,
  rounded predicate, so it is not used.

```python
def _count_1d_pairs_within(v, r):             # #{p < q : |v_p - v_q| <= r}
    s = np.sort(v)
    n = s.size
    p = np.arange(n)
    lo, hi = p + 1, np.full(n, n)
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        mid = (lo + hi) // 2
        ok = (mid < n) & (s[np.minimum(mid, n - 1)] - s <= r)
        lo = np.where(ok & (lo < hi), mid + 1, lo)
        hi = np.where(~ok & (lo < hi), mid, hi)
    return int((lo - (p + 1)).sum())
```

- **A for m = 1.** Count the pairs `i < j` of 2-D points `(x_i, x_{i+1})` within `r` in both coordinates:
  - `numba` path: `dominance_count_self`, a sweep in x-order. It evicts points with `a_p − a_lo > r`, queries a
    Fenwick tree over b-ranks for the rank range found by predicate bisection on sorted b, then inserts `p`.
    Complexity O(N log N).
  - numpy reference: `(cKDTree(pts).count_neighbors(cKDTree(pts), r, p=np.inf) − N) // 2`.
- **General m.** For `m ≥ 2`, both counts use the `cKDTree` path. Tier A fixes `m = 1`; the general path exists
  because `CoordinationParams` accepts `sampen_m ≥ 1`.
- **`cross_sampen`.**
  - B: for each `u_i`, the count of `v_j` within `r`, by two predicate bisections on sorted v.
  - A: `dominance_count_cross`, or the reference `cKDTree(u_pts).count_neighbors(cKDTree(v_pts), r, p=np.inf)`.

- [ ] **Step 1: Write the failing tests.**
  - Cluster:
    - `test_identical_phases_rho_one`: every ρ equals 1 within `1e-12`.
    - `test_constant_per_player_lags_rho_group_one`: each player carries a fixed lag (Frank & Richardson);
      `rho_group_mean` is within `1e-12` of 1, `phi_bar` recovers the lags relative to q, and `rho_k == 1`.
    - `test_uniform_random_phases_rho_small`: 11 players, seeded; `rho_group_mean < 0.45`, which is well below the
      coupled value.
    - `test_min_players_both_sides`: at `n = min_players` the sample is usable; at `min_players − 1` it is not.
    - `test_invalid_players_excluded_per_sample`.
  - Entropy:
    - `test_sampen_gaussian_white_noise_matches_published_analytic_value`: Richman & Moorman 2000's random-number
      case. The expectation is `−ln erf(r / (2σ))` = `−ln erf(0.1)` = 2.185 for `r = 0.2σ`; N = 20,000 seeded is
      within 0.03 of it.
    - `test_sampen_periodic_series_is_low` (< 0.5).
    - `test_entropy_undefined_reachable`: a series with no length-2 matches gives `(nan, 0, B)`.
    - `test_cross_sampen_direction_independent`: `cross_sampen(u, v) == cross_sampen(v, u)` (same value, A and B).
    - `test_counters_exact_against_naive`: the O(N²) double loop over the exact predicate, on 400 continuous
      points **and** 400 tie-heavy integer points; `_count_1d_pairs_within` is equal on both.
  - `test_numba_parity.py` (`pytest.importorskip("numba")`):
    - `dominance_count_self` and `dominance_count_cross` equal the naive counter on the continuous and tie-heavy
      sets. They also equal the `cKDTree` reference on the continuous set, where no pair sits exactly at `r`.
    - `test_force_numpy_env_disables_numba` (monkeypatch the env var).
  - `test_numba_cache_key_covers_all_njit_files` in `tests/test_ci_shard_wiring.py` goes **red** as soon as
    `_numba.py` defines `@njit`. That is the intended detection.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement.**
  - The Fenwick sweep in `_numba.py` uses explicit signatures (`@njit("int64(float64[:], float64[:], float64)",
    cache=True)`), runs serially and has no `prange`.
  - `_entropy.py` dispatches on `use_numba()`. Results are integer counts, so they are identical either way.
  - In **both** cache steps of `.github/workflows/ci.yml` (`test` and `slow`), extend the key to
    `hashFiles('silly_kicks/tracking/**/*_numba*.py', 'silly_kicks/xtgk/_turnover.py',
    'silly_kicks/coordination/**/_numba*.py')`, and update the comment above it (§7.15, §10).
- [ ] **Step 4: Run** the three test files plus `tests/test_ci_shard_wiring.py` → **PASS**, with numba installed
  (`.venv` has numba 0.67.0). Also run them once with `SILLY_KICKS_COORDINATION_FORCE_NUMPY=1` → **PASS**. Then ruff +
  format + pyright.

---

### Task 13: Surrogates (`_surrogates.py`, `_numba.py` part 2)

**Files:**
- Create: `silly_kicks/coordination/_kernels/_surrogates.py`
- Modify: `silly_kicks/coordination/_kernels/_numba.py`
- Test: `tests/coordination/kernels/test_surrogates.py`, `tests/coordination/kernels/test_numba_parity.py` (extend)

**Interfaces — produces:**

```python
def key_words(key: tuple[object, ...]) -> tuple[int, int, int, int]
    # blake2b(digest_size=16) over json.dumps([str(canonical_id(k)) for k in key]) -> 4 little-endian uint32 words
def surrogate_rng(seed: int, key: tuple[object, ...]) -> np.random.Generator
    # np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=key_words(key)))
def shift_bounds(n: int, tau: int) -> tuple[int, int] | None   # (tau, n - tau) when n >= 2 * tau + 1, else None
def draw_shifts(rng: np.random.Generator, n: int, tau: int, k: int) -> np.ndarray | None   # uniform ints in bounds
def iaaft(x: np.ndarray, rng: np.random.Generator, max_iter: int) -> tuple[np.ndarray, bool]  # (surrogate, converged)
def percentile_rank(obs: float, surr: np.ndarray) -> float
    # (#{s < obs} + 0.5 * #{s == obs}) / K over finite surr; NaN if obs is NaN or K == 0
def surrogate_triple(obs: float, surr: np.ndarray) -> tuple[float, float, float]   # (mean, percentile, excess = obs - mean)
# accelerated statistics -- each parity-tested against the direct computation (§7.9 estimator identity)
def shifted_phasor_sums(za: np.ndarray, zb: np.ndarray, shifts: np.ndarray,
                        start: int | None = None, end: int | None = None) -> np.ndarray
    # S[s] = sum_{t in [start, end)} za[t] * conj(zb[(t - s) mod N]); FFT when [start, end) is the whole segment (C11)
def shifted_near_in_phase_counts(za, zb, shifts, start: int, end: int, cos_thr: float) -> np.ndarray   # int64 (K,)
def shifted_lagged_pearson(a: np.ndarray, b: np.ndarray, shifts: np.ndarray, max_lag: int
                           ) -> tuple[np.ndarray, np.ndarray]   # r (K, 2L+1), n (2L+1,); whole-segment slices only
# _numba.py additions
def near_in_phase_counts_nb(za_re, za_im, zb_re, zb_im, shifts, start, end, cos_thr) -> np.ndarray   # int64
def xcorr_wrap_correction_nb(a, b, shifts, max_lag) -> np.ndarray                                   # float64 (K, 2L+1)
```

The algebraic identities, which are the non-obvious part:
- **Phasor sums.** `S[s] = Σ_t za[t]·conj(zb[t−s]) = ifft(fft(za) · conj(fft(zb)))[s mod N]`. One FFT pair gives all
  N shifts; the draws then index into it. Sub-segment windows gather `zb[(np.arange(start, end)[None, :] −
  shifts[:, None]) % N]` in chunks of 32 shifts (C11).
- **Cross-correlation of a circularly shifted `b` over the whole segment.**
  - Let `C(d) = Σ_{t=0}^{N−1} a[t]·b[(t+d) mod N]`, which is `irfft(conj(rfft(a))·rfft(b))[d mod N]`.
  - For shift `s` and lag `ℓ`, the linear overlap sum is `Sab_s(ℓ) = C(ℓ − s) − W_s(ℓ)`.
  - `W_s(ℓ)` is the exact sum of the `|ℓ|` wrapped terms: `t ∈ [N−ℓ, N)` for `ℓ ≥ 0`, `t ∈ [0, −ℓ)` for `ℓ < 0`, each
    `a[t]·b[(t+ℓ−s) mod N]`.
  - The overlap sums of `a` are shift-free prefix sums. Those of `b_s` are **circular** prefix sums over the
    contiguous circular range `[(start+ℓ−s) mod N, …)`.
  - `W` is accumulated **sequentially** in ascending `t`: numpy through `np.cumsum(..., axis=-1)[..., -1]`, and
    `numba` by a plain loop. Both paths then add in the same order, so their parity is exact, not just close
    (§7.15 "float sums follow the reference order").
- **Estimator identity.** Every accelerated statistic feeds the exact functions the observed value uses
  (`circular_summary`, `near_in_phase` counting, the Pearson formula of Task 9, `xcorr_summary`).

`iaaft` (Schreiber & Schmitz 2000):

```python
def iaaft(x, rng, max_iter):
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))
    s = rng.permutation(x)
    ranks = np.argsort(np.argsort(s))
    for _ in range(max_iter):
        spec = np.fft.rfft(s)
        s = np.fft.irfft(target_amp * np.exp(1j * np.angle(spec)), n=x.size)
        new_ranks = np.argsort(np.argsort(s))
        s = sorted_x[new_ranks]
        if np.array_equal(new_ranks, ranks):
            return s, True
        ranks = new_ranks
    return s, False                                   # non-convergence is reported (report counter), never hidden
```

- [ ] **Step 1: Write the failing tests** (§9.1 surrogate bullets):
  - `test_time_shift_preserves_autocorrelation_exactly`: the circular autocorrelation of `np.roll(x, s)` equals that
    of `x` within `1e-12` for every lag.
  - `test_shift_bounds_both_sides`: `n = 2τ+1` gives `(τ, τ+1)`; `n = 2τ` gives `None` (R1 `segment_too_short`).
  - `test_draws_within_bounds`.
  - `test_seed_independent_of_processing_order`: draws for keys k1 and k2 in either order, and in separate
    processes simulated by fresh generators, are identical.
  - `test_key_words_stable_across_id_dtypes`: `key_words((1, 2, "centroid_x"))` equals
    `key_words(("1", 2, "centroid_x"))` (canonical ids).
  - `test_coupled_pair_percentile_separates_from_uncoupled` (non-vacuity):
    - A coupled sinusoid pair gives an R percentile ≥ 0.99.
    - Independent AR(1) processes give a percentile in `[0.05, 0.95]`.
  - `test_accelerated_R_equals_direct`: whole-segment FFT sums and sub-window gathers both equal `np.roll`-then-sum
    within `1e-9`.
  - `test_accelerated_near_in_phase_equals_direct`: exact integers.
  - `test_accelerated_xcorr_equals_direct`: `shifted_lagged_pearson` equals `lagged_pearson(a, np.roll(b, s))` for
    every draw, within `1e-9`.
  - `test_percentile_formula_with_ties`: `obs` equal to 3 of 10 surrogates, 4 below → (4 + 1.5)/10.
  - `test_iaaft_preserves_amplitude_distribution_and_spectrum`:
    - The sorted values are equal exactly.
    - The spectrum amplitude has a relative error < 5% at convergence.
    - The converged flag is reported.
  - `test_fft_and_phasor_calls_constant_in_k` (structural, §9.4):
    - Monkeypatch-count `numpy.fft.fft`/`ifft`/`rfft`/`irfft` as referenced from the `_surrogates` module with
      `tests/_perf_structural.call_counter`.
    - The count for K = 19 equals the count for K = 199 on a whole-segment window.
  - In `test_numba_parity.py`: `near_in_phase_counts_nb` equals the numpy path exactly, and
    `xcorr_wrap_correction_nb` equals the sequential numpy reference **exactly** (`np.testing.assert_array_equal`).
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement.** Import FFTs as `from numpy import fft as _fft`, so the structural counter can wrap
  `_surrogates._fft`.
- [ ] **Step 4: Run** both files with and without `SILLY_KICKS_COORDINATION_FORCE_NUMPY=1` → **PASS**. Then ruff +
  format + pyright.

---

## Package plumbing (Tasks 14–17)

### Task 14: Schemas, vocabularies, params, report (`_columns.py`, `_config.py`, `_provider_params_generated.py`, `_report.py`)

**Files:**
- Create:
  - `silly_kicks/coordination/_columns.py`
  - `silly_kicks/coordination/_config.py`
  - `silly_kicks/coordination/_provider_params_generated.py`
  - `silly_kicks/coordination/_report.py`
- Test:
  - `tests/coordination/test_columns.py`
  - `tests/coordination/test_config.py`
  - `tests/coordination/test_report.py`

**Interfaces — produces.** Every later task imports these names. They are single-sourced here, and every gate iterates
them.

```python
# _columns.py -- vocabularies (tuples, closed)
COORD_LEVELS = ("team_team", "cross_variable", "intra_team", "dyad")
DEFAULT_LEVELS = COORD_LEVELS
COORD_WINDOW_KINDS = ("period", "sliding", "possession")
COORD_WINDOW_SOURCES = ("period", "possession_events", "possession_tracking", "caller")
COORD_AXES = ("x", "y", "scalar", "mixed")                               # C7 (+ "mixed" for a caller's x-vs-y pair)
COORD_METHOD_FAMILIES = ("relative_phase", "cross_correlation", "vector_coding", "coherence",
                         "spectral", "cluster", "team_sync", "rsi")
COORD_SOURCE_VALUES = ("scored", "too_short", "insufficient_detection", "insufficient_players", "goal_end_unresolved",
                       "not_commensurate", "no_possession_role", "degenerate_constant", "entropy_undefined")   # §7.13
COORD_SURROGATE_SOURCE_VALUES = ("computed", "computed_nonconverged", "disabled", "segment_too_short",
                                 "not_scored")                                                  # R1 + amendment A1
COORD_DETECTION_SOURCE_VALUES = ("fully_observed", "detection_aware")
COORD_STOPPAGE_SOURCE_VALUES = ("ball_state", "events", "unavailable")

@dataclass(frozen=True)
class SignalSpec:
    unit: Literal["metres", "m^2", "dimensionless"]
    kind: Literal["positional", "magnitude", "binary"]
    axis: Literal["x", "y", "scalar"]
    scope: Literal["team", "player", "match"]

COORD_SIGNALS: Mapping[str, SignalSpec]     # MappingProxyType; exactly these 15 entries:
#   centroid_x (metres, positional, x, team)      centroid_y (metres, positional, y, team)
#   team_length (metres, magnitude, x, team)      team_width (metres, magnitude, y, team)
#   stretch_index (metres, magnitude, scalar, team)
#   stretch_x (metres, magnitude, x, team)        stretch_y (metres, magnitude, y, team)
#   spread (metres, magnitude, scalar, team)      convex_hull_area (m^2, magnitude, scalar, team)
#   defensive_line_x (metres, positional, x, team) compactness_x (metres, magnitude, x, team)
#   back_line_high_x (metres, positional, x, team)
#   player_x (metres, positional, x, player)      player_y (metres, positional, y, player)
#   possession (dimensionless, binary, scalar, match)
TEAM_SIGNALS = tuple(s for s, spec in COORD_SIGNALS.items() if spec.scope == "team")   # 12, the spectral set
```

**Output schemas** are plain `dict[str, str]` name → dtype, in emitted order. The dtype vocabulary is `object`,
`int64`, `Int64` and `float64` (the TERRITORY_COLUMNS precedent).

C22 (concretisation) covers the id columns:
- The pair keys `team_a_id`, `team_b_id`, `player_a_id` and `player_b_id` are canonical ids (§7.12 "`canonical_id`
  in pair keys"), declared `object`.
- The single-entity ids `game_id`, `team_id` and `player_id` are `restore_id_dtype`-restored to the source frames'
  dtype (§7.12). They are declared `object`, meaning "id-valued", as `TERRITORY_COLUMNS` does.
- `period_id` is `int64`, and `window_id` and `phase_index` are `Int64`.

| Constant | Content (in order) |
|---|---|
| `COORD_WINDOW_COLUMNS` | `game_id, period_id, window_kind, window_id, window_source, start_time_s, end_time_s, attacking_team_id, terminal_action, terminal_team_id, n_phases` |
| `COORD_PAIR_KEYS` | `game_id, period_id, window_kind, window_id, level, signal_a, signal_b, axis, team_a_id, team_b_id, player_a_id, player_b_id` |
| `COORD_PAIR_METRIC_COLUMNS` | `RP` + `XC` + `VC` + `COH` + `COVERAGE` (below) |
| `COORD_PAIR_COLUMNS` | keys + metric + `coord_rp_source, coord_xc_source, coord_vc_source, coord_coh_source` + (R1) `coord_rp_surrogate_source, coord_xc_surrogate_source, coord_vc_surrogate_source, coord_coh_surrogate_source` + `coord_detection_source, coord_stoppage_source` |
| `COORD_PAIR_PHASE_KEYS` | `COORD_PAIR_KEYS + ("phase_index",)` |
| `COORD_PAIR_PHASE_METRIC_COLUMNS` | `RP_CORE + RP_HIST + VC_CORE + ("coord_duration_s", "coord_n_samples")` (C9) |
| `COORD_PAIR_PHASE_COLUMNS` | keys + metric + `coord_rp_source, coord_vc_source, coord_detection_source, coord_stoppage_source` |
| `COORD_SPECTRAL_KEYS` | `game_id, period_id, window_kind, window_id, team_id, signal` |
| `COORD_SPECTRAL_METRIC_COLUMNS` | `coord_median_freq_cpm, coord_duration_s, coord_n_segments` |
| `COORD_SPECTRAL_COLUMNS` | keys + metric + `coord_spectral_source, coord_detection_source, coord_stoppage_source` |
| `COORD_CLUSTER_TEAM_KEYS` | `game_id, period_id, window_kind, window_id, team_id, axis` |
| `COORD_CLUSTER_TEAM_METRIC_COLUMNS` | `coord_rho_group_mean, coord_rho_group_sd, coord_rho_group_sampen, coord_n_players_mean, coord_rho_group_mean_surrogate_mean, coord_rho_group_mean_percentile, coord_rho_group_mean_excess, coord_duration_s, coord_n_samples, coord_observed_fraction` |
| `COORD_CLUSTER_TEAM_COLUMNS` | keys + metric + `coord_cluster_source, coord_cluster_surrogate_source` (R1) `, coord_detection_source, coord_stoppage_source` |
| `COORD_CLUSTER_PLAYER_KEYS` | `COORD_CLUSTER_TEAM_KEYS[:5] + ("player_id", "axis")` |
| `COORD_CLUSTER_PLAYER_METRIC_COLUMNS` | `coord_phi_mean_deg, coord_rho_k, coord_phi_sd_deg, coord_phi_sampen, coord_on_pitch_s` |
| `COORD_CLUSTER_PLAYER_COLUMNS` | keys + metric + `coord_cluster_player_source, coord_detection_source, coord_stoppage_source` |
| `COORD_TEAM_SYNC_KEYS` | `game_id, period_id, window_kind, window_id, axis` |
| `COORD_TEAM_SYNC_METRIC_COLUMNS` | `coord_team_sync_pearson_r, coord_team_sync_cross_sampen, coord_team_sync_pearson_r_surrogate_mean, coord_team_sync_pearson_r_percentile, coord_team_sync_pearson_r_excess` |
| `COORD_TEAM_SYNC_COLUMNS` | keys + `team_a_id, team_b_id` (C8) + metric + `coord_team_sync_source, coord_team_sync_surrogate_source, coord_detection_source, coord_stoppage_source` |
| `COORD_RSI_KEYS` | `game_id, period_id, window_kind, window_id, axis` |
| `COORD_RSI_METRIC_COLUMNS` | `coord_rsi_mean_m, coord_rsi_fraction_positive, coord_rsi_switch_rate_per_min, coord_rsi_bimodality_coefficient` |
| `COORD_RSI_COLUMNS` | keys + `team_a_id, team_b_id` (C8) + metric + `coord_rsi_source, coord_detection_source, coord_stoppage_source` |

The metric groups are exact tuples:

```python
RP_CORE = ("coord_rp_mean_deg", "coord_rp_circ_sd_deg", "coord_rp_resultant_length", "coord_rp_pct_near_in_phase")
RP_HIST = tuple(f"coord_rp_hist_bin_{lab}" for lab in HIST_BIN_LABELS)          # 12, from _kernels._circular
RP_VALID = ("coord_rp_phase_valid_fraction_a", "coord_rp_phase_valid_fraction_b")
def _triple(m): return (f"{m}_surrogate_mean", f"{m}_percentile", f"{m}_excess")
RP = RP_CORE + RP_HIST + RP_VALID + _triple("coord_rp_resultant_length") + _triple("coord_rp_pct_near_in_phase")
XC = ("coord_xc_max_abs_r", "coord_xc_lag_s", "coord_xc_r_at_max", "coord_xc_r_lag0") + _triple("coord_xc_max_abs_r")
VC_CORE = ("coord_vc_pct_in_phase", "coord_vc_pct_anti_phase", "coord_vc_pct_a_phase", "coord_vc_pct_b_phase",
           "coord_vc_mean_angle_deg", "coord_vc_angle_variability_deg", "coord_vc_n_stationary")
VC = VC_CORE + _triple("coord_vc_pct_in_phase") + _triple("coord_vc_pct_anti_phase")
COH = ("coord_coh_band_mean", "coord_coh_peak_freq_cpm", "coord_coh_n_segments") + _triple("coord_coh_band_mean")
COVERAGE = ("coord_duration_s", "coord_n_samples", "coord_n_segments", "coord_observed_fraction_a", "coord_observed_fraction_b")
```

- **Value types.** `pct`/`fraction`/`percentile` values are fractions in [0, 1], the repo's `_pct` = ratio convention
  (e.g. `field_tilt_pct`). Counts (`coord_n_*`, `coord_coh_n_segments`, `coord_vc_n_stationary`) are `Int64`; every
  other metric is `float64`.
- **Contract families.** Seven are exported: `coordination_pair`, `coordination_pair_phase`,
  `coordination_spectral`, `coordination_cluster_team`, `coordination_cluster_player`, `coordination_team_sync` and
  `coordination_rsi`. Each has `*_KEYS`, `*_METRIC_COLUMNS` and a `*_COLUMNS` dict (§7.12).

```python
# _config.py
@dataclass(frozen=True)
class CoordinationParams:
    # Tier A -- primary sources, never tuned (§8.1)
    xcorr_max_lag_s: float = 15.0
    n_phases: int = 3
    near_in_phase_deg: float = 30.0
    max_stoppage_s: float = 25.0
    sampen_m: int = 1
    sampen_r_sd: float = 0.2
    butterworth_order: int = 3
    analysis_hz: float = 10.0
    # Conventions -- fixed, with rationale (§8.1)
    min_players: int = 6
    n_surrogates: int = 199
    iaaft_max_iter: int = 100
    coverage_warn_fraction: float = 0.25
    # Tier B -- base values SINGLE-SOURCED from the generated file (amendment A3). No literal lives here:
    # _BASE = _provider_params_generated.BASE_COORDINATION_PARAMS
    # (commit 1: BASE_SOURCE == "interim" = the R4 table; commit 2: "derivation" = D1's pooled values)
    butterworth_cutoff_hz: float = _BASE["butterworth_cutoff_hz"]
    max_detection_gap_s: float = _BASE["max_detection_gap_s"]
    min_observed_fraction: Mapping[str, float] = field(default_factory=lambda: _base_map("min_observed_fraction"))
        # keys: every COORD_METHOD_FAMILIES entry
    vc_epsilon: Mapping[str, float] = field(default_factory=lambda: _base_map("vc_epsilon"))
        # keys: every TEAM_SIGNALS entry
    min_shift_s: Mapping[str, float] = field(default_factory=lambda: _base_map("min_shift_s"))
        # keys: TEAM_SIGNALS + player_x, player_y + "cluster_amplitude"
    band_low_cpm: float = _BASE["band_low_cpm"]
    band_high_cpm: float = _BASE["band_high_cpm"]
    welch_segment_s: float = _BASE["welch_segment_s"]
    possession_gap_s: float = _BASE["possession_gap_s"]
    # Other
    surrogate_method: Literal["time_shift", "iaaft"] = "time_shift"
    surrogate_seed: int = 0
    include_goalkeeper: Mapping[str, bool] = field(default_factory=_default_include_goalkeeper)
        # {"team_signals": False (Moura: outfield), "dyad": False (Folgado: outfield), "cluster": True (Duarte: 11 incl. GK)}
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    def __post_init__(self) -> None: ...          # C16: freeze maps (MappingProxyType), assert exact key sets, reject:
        # n_phases < 1; xcorr_max_lag_s <= 0; near_in_phase_deg not in (0, 180); max_stoppage_s <= 0; sampen_m < 1;
        # sampen_r_sd <= 0; butterworth_order < 1; analysis_hz <= 0; butterworth_cutoff_hz <= 0; min_players < 2;
        # n_surrogates < 0; iaaft_max_iter < 1; coverage_warn_fraction not in [0, 1]; max_detection_gap_s < 0;
        # min_observed_fraction value not in [0, 1]; vc_epsilon value < 0; min_shift_s value <= 0;
        # not 0 < band_low_cpm < band_high_cpm; welch_segment_s <= 0; possession_gap_s < 0; surrogate_seed < 0
    def __hash__(self) -> int: ...                # hash of a canonical tuple (maps as sorted items)
    @classmethod
    def default(cls, *, force_universal: bool = False) -> CoordinationParams: ...   # RestDefenseParams idiom
    @classmethod
    def for_provider(cls, provider: str) -> CoordinationParams: ...
        # base merged with _provider_params_generated.PROVIDER_COORDINATION_PARAMS.get(provider, {}); map fields
        # merge KEY-WISE
    def is_default(self) -> bool: ...
```

`_provider_params_generated.py` holds plain data and imports nothing from the package, so there is no import cycle.
In commit 1 it is a generated-file header plus:
- `BASE_SOURCE: str = "interim"`;
- `BASE_COORDINATION_PARAMS: dict[str, object]` = the R4 table, with complete maps for the three map fields;
- `PROVIDER_COORDINATION_PARAMS: dict[str, dict[str, object]] = {}`.

Task 19's codegen renders exactly this for empty inputs (A3). `_config.py` defines `_base_map(name)`, which returns a
fresh `dict` copy of `BASE_COORDINATION_PARAMS[name]` (then frozen by `__post_init__`).

```python
# _report.py
class CoordinationCoverageWarning(UserWarning): ...   # subclasses no other category (§7.13)

@dataclass(frozen=True)
class CoordinationReport:
    params: CoordinationParams
    provider: str
    window_source: str
    detection_source: str
    stoppage_source: str
    native_hz: float
    effective_hz: float
    rate_capped: bool
    n_windows_in: int
    n_windows_scored: int
    windows_dropped: Mapping[str, int]                    # reason token -> windows with NO scored row in any method
    n_segments: int
    n_runs_too_short: int
    samples_stationary: int
    samples_unobserved: int
    samples_below_min_players: int
    rows_by_source: Mapping[str, Mapping[str, int]]       # family -> COORD_SOURCE_VALUES token -> rows
    surrogate_rows_by_source: Mapping[str, Mapping[str, int]]   # family -> COORD_SURROGATE_SOURCE_VALUES token (R1)
    dead_seconds: float
    n_stoppage_splits: int
    iaaft_nonconverged: int
    def merge(self, other: CoordinationReport) -> CoordinationReport: ...
        # the provenance fields must be equal (else ValueError); family maps are disjoint unions; sample counters sum
    def conservation_errors(self) -> list[str]: ...
        # [] iff n_windows_in == n_windows_scored + sum(windows_dropped.values()) and every family's token rows sum
        # equals that family's row count carried alongside
```

The drop reason for a window with no scored row is the **first** token in this precedence:
`goal_end_unresolved, insufficient_detection, insufficient_players, too_short, no_possession_role, not_commensurate,
degenerate_constant, entropy_undefined`. That makes the attribution deterministic.

- [ ] **Step 1: Write the failing tests.**
  - `test_columns.py`:
    - Every `*_COLUMNS` dict's keys start with its `*_KEYS` in order.
    - `set(metric) ⊆ set(columns)`.
    - Metric and provenance are disjoint, and every non-key, non-metric column ends in `_source` or is
      `team_a_id`/`team_b_id`.
    - All metric names start with `coord_`.
    - The vocabularies are tuples with no duplicates.
    - `COORD_SIGNALS` has exactly 15 entries with the units, kinds, axes and scopes listed.
    - `len(RP_HIST) == 12`.
    - `test_counts_are_nullable_int`.
  - `test_config.py`:
    - `test_tier_a_and_convention_defaults` (every literal above).
    - `test_tier_b_defaults_come_from_generated_base` (A3): every Tier-B field default equals
      `BASE_COORDINATION_PARAMS`, and an AST check proves `_config.py` contains no numeric literal for a Tier-B field.
    - `test_interim_base_matches_R4_in_commit_1`: `BASE_SOURCE == "interim"` and the values equal the R4 table. Task 28
      replaces this test.
    - `test_maps_complete_and_frozen`: a key set other than the declared one raises, and assigning into a map raises
      `TypeError`.
    - `test_every_rejection_both_sides`: one parametrised case per rejection above, at the boundary value (rejected)
      and one step inside (accepted).
    - `test_for_provider_merges_keywise`: monkeypatch the generated map with a partial `vc_epsilon` override; the
      untouched keys keep their base value.
    - `test_for_provider_unlisted_is_base`.
    - `test_default_flag_idiom` (`default().is_default()`, `force_universal`, and a hand-built config is not
      default).
    - `test_hashable_and_equal_hash_for_equal_params`.
    - `test_generated_map_empty_in_commit_1`. Task 28 rewrites this test to the generated content.
  - `test_report.py`:
    - `test_warning_category_is_distinct`: `CoordinationCoverageWarning` is not a subclass of `SyntheticEPVWarning`,
      `IgnoredSurfaceInputsWarning` or `OrientationUnresolvedWarning`, and they are not subclasses of it.
    - `test_merge_sums_and_rejects_provenance_mismatch`.
    - `test_conservation_errors_detects_a_planted_leak`.
    - `test_drop_reason_precedence`.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement** to the interfaces. Every public class and function carries an Examples section:
  - doctests for `CoordinationParams.default/for_provider/is_default`, `PairSpec`-free constants and
    `CoordinationReport.conservation_errors`;
  - an RST literal block where a doctest would need frames.
- [ ] **Step 4: Run** → **PASS**. Then ruff + format + pyright.

---

### Task 15: Windows and stoppage evidence (`_windows.py`)

**Files:**
- Create: `silly_kicks/coordination/_windows.py`
- Create: `tests/coordination/_fixtures.py` (the synthetic match generator, shared by Tasks 15–18)
- Test: `tests/coordination/test_windows.py`

**Interfaces — produces:**

```python
RESTART_TYPES = ("throw_in", "freekick_crossed", "freekick_short", "corner_crossed", "corner_short",
                 "goalkick", "shot_freekick", "shot_penalty")                        # §7.6 (D20)
SHOT_TYPES = ("shot", "shot_freekick", "shot_penalty")
def period_windows(frames, *, length_s: float | None = None, step_s: float | None = None) -> pd.DataFrame
def possession_windows_from_actions(actions, frames, *, links=None, n_phases: int = 3,
                                    possession_kwargs: Mapping[str, Any] | None = None) -> pd.DataFrame
def possession_windows_from_frames(frames, *, carrier=None, n_phases: int = 3,
                                   params: CoordinationParams | None = None) -> pd.DataFrame
def validate_windows(windows: pd.DataFrame) -> str
    # -> the call's window "regime"; raises on a contract violation or a forbidden source mix (C25)
@dataclass(frozen=True)
class StoppageEvidence:
    source: Literal["ball_state", "events", "unavailable"]
    intervals: Mapping[tuple[object, object], np.ndarray]   # (canonical game, period) -> (S, 2) dead [start, end) > max
    dead_seconds: float                                      # total length of the splitting intervals
    n_splits: int
def resolve_stoppages(frames, *, actions, provider: str, max_stoppage_s: float,
                      mode: Literal["auto", "ball_state", "events", "none"] = "auto") -> StoppageEvidence
def phase_assignment(m: int, n_phases: int) -> np.ndarray        # C5: k = ((i+1)*n + m - 1) // m, i = 0..m-1 -> 1..n
```

Builder semantics (spec §7.6 plus the concretisations named here):
- **`period_windows`.**
  - One window per `(game, period)`: `start = min(time_seconds)`, `end = max(time_seconds) + 1/frame_rate`.
  - With `length_s`/`step_s`, the windows are `[start + j·step, start + j·step + length)`, **full-length only**
    (C21). They are `window_kind="sliding"`, `window_source="period"`.
  - `window_id` counts from 0 per `(game, period, kind)` by start time.
  - `attacking_team_id`, `terminal_action` and `terminal_team_id` are `<NA>`; `n_phases` is `<NA>`.
- **`possession_windows_from_actions`.** Vectorised, with no per-possession Python loop:
  - Possessions come from `spadl.add_possessions(actions, **(possession_kwargs or {}))`. The attacking team is the
    team of the possession's first action.
  - The window ends at the event that **ends** the possession: the attacking team's last action if it is a
    `SHOT_TYPES` type, otherwise the first action of the next possession.
  - If the next possession is in another period, or absent, the window ends at the period's last frame time and the
    terminal columns are `<NA>` (the possession ended with the period; C21).
  - Times are the actions' period-relative `time_seconds`.
  - **C23:** with `links` given, a linked action's start and end times are its linked frame's `time_seconds`
    (unlinked actions keep their action time), so a caller-supplied linkage is reused as §7.6 says.
  - `window_source = "possession_events"`.
- **`possession_windows_from_frames`.**
  - Possession comes from `derive_team_in_possession(frames, carrier or infer_ball_carrier(frames))`, per frame.
  - An NA gap of at most `params.possession_gap_s` between two frames of the **same** team is bridged. Any other gap
    ends the spell.
  - Spells are maximal runs of one team: `start = first frame time`, `end = last frame time + 1/frame_rate`.
  - The terminal columns are `<NA>`. `window_source = "possession_tracking"`.
- **C25 — the source-mix rule** (§7.3 "windows from mixed sources" read against D3, "events and tracking are never
  mixed"):
  - `period` windows may accompany **one** possession source;
  - `possession_events` together with `possession_tracking` → `ValueError`;
  - `caller` windows cannot mix with any builder source.

  This lets the orchestrator's default (§7.2: period windows plus one possession builder) pass the contract.
- **`resolve_stoppages`:**
  - `ball_state` needs `dead_ball_observed(provider)`. Its intervals are the maximal runs of frames whose ball row
    has `ball_state == "dead"`, `[first dead frame time, first alive frame time after)`.
  - `events` needs actions. It builds one interval from the previous action's `time_seconds` to each `RESTART_TYPES`
    action's `time_seconds`, and one from each goal to the next action's time. A goal is a `SHOT_TYPES` action with
    result `success`, or any action with result `owngoal`. Only same-period pairs count.
  - Only intervals **strictly longer** than `max_stoppage_s` are kept (both sides tested).
  - `auto` is the D20 precedence: ball_state, then events, then unavailable.
  - An explicit mode that cannot be honoured (for example `ball_state` on skillcorner) raises `ValueError` (C12).

- [ ] **Step 1: Write `tests/coordination/_fixtures.py`.** Keep it in the style of `tests/restdefense/_fixtures.py`.
  It is deterministic and generates `TRACKING_FRAMES_COLUMNS`-conformant frames:

```python
def make_coordination_match(
    *, seconds: float = 900.0, hz: float = 10.0, provider: str = "sportec", n_outfield: int = 10,
    with_gk: bool = True, oscillation_cpm: float = 0.5, phase_offset_deg: float = 40.0, noise_m: float = 0.3,
    dead_intervals: Sequence[tuple[float, float]] = (), visibility_drop: float = 0.0,
    substitution: tuple[float, int] | None = None, red_card: tuple[float, int] | None = None,
    team_ids: tuple[object, object] = (1, 2), seed: int = 58, periods: int = 1,
) -> pd.DataFrame
    # Team centroids oscillate longitudinally at oscillation_cpm, team B lagging by phase_offset_deg; players
    # jitter around formation slots; team 1 attacks LTR in period 1 (GK near x=4), team 2 RTL (GK near x=101);
    # ball_state "dead" inside dead_intervals (else "alive"); visibility False on a seeded fraction of outfield
    # samples when provider == "skillcorner"; a substitution swaps a player id at t without splicing; a red card
    # removes a player at t.
def make_coordination_actions(frames, *, restarts: Sequence[tuple[float, str]] = (), goals: Sequence[float] = (),
                              possession_every_s: float = 12.0) -> pd.DataFrame
    # Minimal SPADL actions (game_id, period_id, action_id, time_seconds, team_id, player_id, type_id, result_id,
    # start_x, start_y, end_x, end_y, bodypart_id) with alternating possessions, the given restarts and goals.
```

  Also write `test_fixture_preconditions` in `test_windows.py`, following ADR-032:
  - the generated match has the planted oscillation (the team-centroid spectral peak lies within one bin of
    `oscillation_cpm`);
  - both directions are present;
  - the dead intervals are present in `ball_state`.
- [ ] **Step 2: Write the failing window tests.**
  - `test_period_windows_one_per_period_and_sliding_full_length_only`.
  - `test_window_contract_columns_and_dtypes` (all three builders).
  - `test_possession_windows_end_at_terminal_event`:
    - a possession ending in a shot ends at the shot, with `terminal_action = "shot"` and the attacking team;
    - a possession ending in an opponent tackle ends at the tackle, with `terminal_team_id` = the opponent;
    - the last possession of a period ends at the period end with `<NA>` terminals.
  - `test_links_override_action_times` (C23).
  - `test_frames_possession_bridges_same_team_gaps_both_sides`: gap = `possession_gap_s` bridges;
    `possession_gap_s + 1/hz` splits; a gap between different teams always splits.
  - `test_mixed_possession_sources_refused` and `test_caller_mixed_with_builder_refused` (C25).
  - `test_caller_windows_validated` (a missing column or a bad token raises).
  - `test_stoppage_ball_state_intervals` (sportec fixture).
  - `test_stoppage_precedence_each_source_reachable`:
    - sportec gives `ball_state`;
    - skillcorner with actions gives `events`;
    - skillcorner without actions gives `unavailable`;
    - a constant-`"alive"` skillcorner frame set **never** reports `ball_state`.
  - `test_event_intervals_for_every_restart_type_and_goals`: parametrised over `RESTART_TYPES`, plus shot-success and
    own-goal goals.
  - `test_only_longer_than_max_stoppage_splits`: an interval of exactly 25.0 s does not split; 25.0 + 1e-6 does.
  - `test_dead_ball_observed_unclassified_provider_raises`.
  - `test_explicit_mode_that_cannot_be_honoured_raises`.
  - `test_phase_assignment_rank_rule`:
    - m=9, n=3 → [1,1,1,2,2,2,3,3,3];
    - m=10, n=3 → k from `((i+1)*3+9)//10` → [1,1,1,2,2,2,3,3,3,3];
    - m=2, n=3 → [2,3];
    - the first sample is always assigned (C5).
- [ ] **Step 3: Run** → **FAIL**.
- [ ] **Step 4: Implement.** No per-window Python loop touches a DataFrame. Group once with `group_rows`, and
  register every caller in `SCALE_GUARDED` in Task 18.
- [ ] **Step 5: Run** → **PASS**. Then ruff + format + pyright.

---

### Task 16: Pair catalog and signal preparation (`_catalog.py`, `_signals.py`)

**Files:**
- Create: `silly_kicks/coordination/_catalog.py`, `silly_kicks/coordination/_signals.py`
- Test: `tests/coordination/test_catalog.py`, `tests/coordination/test_signals.py`

**Interfaces — produces:**

```python
# _catalog.py
@dataclass(frozen=True)
class PairSpec:
    level: str
    signal_a: str
    signal_b: str
    role: Literal["canonical", "attacking_defending", "same_team"]
    # __post_init__: level in COORD_LEVELS; signals in COORD_SIGNALS and != "possession"; allowed (level, role):
    #   team_team {canonical, attacking_defending}; cross_variable {attacking_defending}; intra_team {same_team};
    #   dyad {same_team, canonical} with signal_a == signal_b in {player_x, player_y}; non-dyad levels only
    #   scope == "team" signals.
    @property
    def axis(self) -> str: ...            # common SignalSpec axis, else "mixed"
    @property
    def commensurate(self) -> bool: ...   # units equal (§7.7 vector-coding rule)
DEFAULT_PAIRS: tuple[PairSpec, ...]
    # L1: team_team canonical, signal == signal for the 9 §7.7 signals;
    # L2: cross_variable attacking_defending
    #     (centroid_x, defensive_line_x), (team_length, compactness_x), (stretch_x, stretch_x);
    #     intra_team same_team (defensive_line_x, centroid_x);
    # L3: dyad same_team and canonical for player_x and player_y.
METHODS_BY_LEVEL: Mapping[str, tuple[str, ...]]   # dyad -> (relative_phase, cross_correlation); others -> rp, xc, vc, coh
def resolve_pairs(pairs: Sequence[PairSpec] | None, levels: Sequence[str]) -> tuple[PairSpec, ...]

# _signals.py
@dataclass(frozen=True)
class PlayerSeries:
    team_id: object
    player_id: object
    is_goalkeeper: bool
    x: np.ndarray
    y: np.ndarray                                   # (N,) filtered+resampled, detection-bridged, oriented (C24)
    runs: np.ndarray                                # (R, 2) int [start, end) sample ranges
    phasor_x: np.ndarray
    phasor_y: np.ndarray                            # complex (N,), NaN+NaN*1j outside runs
    inst_freq_pos_x: np.ndarray
    inst_freq_pos_y: np.ndarray                     # bool (N,), phase-validity indicator
    observed: np.ndarray                            # bool (N,)
    on_pitch: np.ndarray                            # bool (N,)

@dataclass(frozen=True)
class PeriodSignals:
    game_id: object
    period_id: int
    t: np.ndarray                                   # (N,) = arange(N) / fs, period-relative
    team_ids: tuple[object, object]                 # canonical-id order; A = team_ids[0]
    goal_x: Mapping[object, float | None]           # defended end per team (GoalMap, allow_guess=True)
    reference_flip: bool                            # C24: True iff team A defends x = 105 (frame was reflected)
    segments: Mapping[object, np.ndarray]           # team -> (S, 2) int [start, end)
    segment_id: Mapping[object, np.ndarray]         # team -> (N,) int, -1 outside segments
    team_signal: Mapping[tuple[object, str], np.ndarray]    # (team, signal) -> (N,) float64
    team_phasor: Mapping[tuple[object, str], np.ndarray]    # complex (N,)
    team_inst_freq_pos: Mapping[tuple[object, str], np.ndarray]
    observed_fraction: Mapping[object, np.ndarray]  # team -> (N,) float64 in [0, 1]
    players: Mapping[tuple[object, object], PlayerSeries]
    possession_team: np.ndarray                     # (N,) object: canonical team id or pd.NA
    window_ranges: np.ndarray                       # (W_p, 3) int64: windows-row index, start, end sample

@dataclass(frozen=True)
class CoordinationSignals:
    provider: str
    params: CoordinationParams
    native_hz: float
    fs: float
    rate_capped: bool
    windows: pd.DataFrame
    window_regime: str
    detection_source: str
    stoppage: StoppageEvidence
    periods: tuple[PeriodSignals, ...]
    counters: Mapping[str, int]                     # n_segments, n_runs_too_short, samples_unobserved, ...

def build_coordination_signals(
    frames: pd.DataFrame, *, windows: pd.DataFrame, params: CoordinationParams | None = None,
    actions: pd.DataFrame | None = None, goal_map: GoalMap | None = None, links: pd.DataFrame | None = None,
    stoppage_evidence: Literal["auto", "ball_state", "events", "none"] = "auto",        # C12
    detection: Literal["auto", "detection_aware", "fully_observed"] = "auto",           # C12
) -> CoordinationSignals
```

**C24 — orientation.** It is applied algebraically, which keeps "phases once per segment" (D15) compatible with
"per-window reference team" (§7.10):
- `_signals` expresses every positional signal (the `COORD_SIGNALS` kind `positional`) in **team A's** goal-relative
  frame, via `to_goal_relative_x_array` / `_y_array` with `goal_x = goal_x[team A]`.
- A window whose reference team is B (possession windows where B attacks) needs team B's frame. That is the 180°
  point reflection of team A's frame, which **negates** every centred positional signal. Its consequences:
  - the Hilbert phase shifts by exactly π (the analytic signal is negated);
  - a Pearson r involving exactly one positional signal flips sign;
  - positional differences are negated (vector coding);
  - coherence, spectra, cluster statistics and every magnitude signal are invariant.
- Task 17 applies these per window through one helper, `_orientation_signs(spec, reference_is_b) -> (s_a, s_b)`, with
  `s ∈ {+1, −1}`.
- `goal_x[team A]` of `None` makes every row needing orientation `goal_end_unresolved`. So does a `None` end for a
  team whose back line is needed (`defensive_line_x`, `back_line_high_x`).

Preparation steps (§7.4). Implement them as private functions in `_signals.py`, each unit-tested:
1. **Refusals, in §7.3's order:**
   - missing columns;
   - `source_provider == "snapshot"`;
   - more than one `source_provider`;
   - unoriented frames (`team_attacking_direction` absent or all null);
   - duplicate `(game, period, frame, player_id)` among non-ball rows;
   - more than one ball row per frame;
   - a detection-aware provider with all-null `visibility` (`assert_detection_aware_visibility`);
   - `validate_provider` for an unclassified provider;
   - windows through `validate_windows`;
   - a `frame_rate` that is not single-valued per call.

   Each raises `ValueError` with the remedy named in §7.3.
2. **Effective rate.** `fs = min(native_hz, max(analysis_hz, 10 · cutoff))`, with `rate_capped = fs < max(analysis_hz,
   10 · cutoff)`. A cutoff design frequency at or above the native Nyquist raises. The check happens before any
   filtering, with the Task 4 message.
3. **Stoppages and detection mode.** `resolve_stoppages(..., mode=stoppage_evidence)`. The detection mode comes from
   `detection`, and `"auto"` means `detection_aware` iff the provider is in `_DETECTION_AWARE_PROVIDERS`.
4. **Goal map.** `goal_map or resolve_defended_goals(frames)`, built once.
5. **Per `(game, period)`, via `group_rows(frames, ("game_id", "period_id"))`:**
   - Dense scatter:
     - frame index = `np.searchsorted(unique_frame_ids, frame_id)`;
     - player slot = `pd.Index(sorted canonical player ids per team).get_indexer(canonical_id_series(player_id))`;
     - coordinates are read with `.to_numpy(dtype="float64")`.
   - Runs: split each player's presence at stoppage intervals, at absences longer than `max_detection_gap_s`, and
     (player-level version only) at undetected stretches longer than `max_detection_gap_s`.
   - Bridging:
     - the player-level version replaces undetected samples inside gaps of at most `max_detection_gap_s` by linear
       interpolation between detections;
     - the team-level version keeps every present row's best-estimate position (§7.11).

     For a fully observed provider the two versions are one array.
   - Filtering: each run is filtered at the native rate by `butterworth_lowpass`. A run shorter than
     `butterworth_min_length` is dropped and counted (`n_runs_too_short`).
   - Resampling: `resample_uniform` onto the period grid `k / fs`, `N = floor(max(time_seconds) · fs) + 1`.
   - Team signals at each sample:
     - Take the on-pitch outfield players (the goalkeeper per `include_goalkeeper["team_signals"]`).
     - The sample is valid iff every on-pitch outfield player has a valid resampled position.
     - Then `compact_rows`, then `collective_from_positions`.
     - The back line comes from `back_line_batch(n=4, adaptive_max_n=5)` (the TF-14 defaults), with
       `defends_x0 = goal_x == 0.0`.
     - `observed_fraction` = detected on-pitch count / on-pitch count; 1.0 for fully observed data.
   - Segments per team: maximal intervals of valid samples with no stoppage and a constant on-pitch outfield count
     (a red card splits; a substitution does not).
   - Possession series: the attacking team of the possession window covering the sample. It holds the last value
     between windows and is `<NA>` before a period's first possession and when the call has no possession windows
     (§7.8.4).
   - Phasors, once per segment or run (D15): `analytic_phase(values, pad_length(len, fs, band_low_cpm))` →
     `phasor`, plus the positive-instantaneous-frequency indicator.
   - `window_ranges` from `np.searchsorted(t, start/end)`.

- [ ] **Step 1: Write the failing tests.**
  - `test_catalog.py`:
    - `test_default_catalog_matches_spec_table`: the counts per level, and each tuple exactly.
    - `test_pairspec_validation_each_rule_both_sides`.
    - `test_axis_and_commensurate_properties` (`convex_hull_area`/`spread` → `commensurate=False`, axis `scalar`;
      `centroid_x`/`centroid_y` → `mixed`).
    - `test_resolve_pairs_filters_levels`.
  - `test_signals.py` (on the Task 15 fixtures):
    - `test_every_refusal` (parametrised over the Step 1 list; each message names its remedy).
    - `test_effective_rate_default_raised_and_capped`: 10 Hz default; cutoff 1.5 Hz on 25 Hz native → 15 Hz; cutoff
      1.5 on 10 Hz native → 10 Hz with `rate_capped=True`; a cutoff at or above Nyquist raises.
    - `test_stoppage_splits_segments_both_sides` (interval 25.0 vs 25.0+ε).
    - `test_detection_gap_split_both_sides` (skillcorner: gap = `max_detection_gap_s` bridged, longer splits).
    - `test_substitution_no_splice`: the replaced and replacing players are separate `PlayerSeries` whose runs do not
      overlap; the team segment does not split at the substitution.
    - `test_red_card_splits_team_segment_and_steps_count`.
    - `test_short_run_dropped_and_counted`.
    - `test_team_signals_equal_collective_kernel`: on the resampled positions, `centroid_x` equals
      `collective_from_positions` on the same samples.
    - `test_observed_fraction`: 1.0 for sportec; equals the planted visibility share within 1/n for skillcorner.
    - `test_possession_series_hold_rule`.
    - `test_orientation_team_a_frame` (C24): positional signals equal the goal-relative transform of team A's end;
      `reference_flip` is set when team A defends 105.
    - `test_phasors_computed_once_per_segment`: `call_counter` on `analytic_phase` equals the number of (series,
      segment) pairs, independent of the window count.
    - `test_no_input_mutation`.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement.** Coordinates are read as float64 at the boundary (ADR-106), and every id goes through
  `id_compat`. Grouping on `team_id`/`player_id` uses `observed=True` (the F1b `category` future, §13.1).
- [ ] **Step 4: Run** → **PASS**. Then ruff + format + pyright.

---

### Task 17: Family computes, orchestrator, raw series (`_compute.py`, `_series.py`)

**Files:**
- Create: `silly_kicks/coordination/_compute.py`, `silly_kicks/coordination/_series.py`
- Test:
  - `tests/coordination/test_compute.py`
  - `tests/coordination/test_series.py`
  - `tests/coordination/test_degradation.py`

**Interfaces — produces (the §7.2 surface):**

```python
def compute_relative_phase(signals, *, levels=DEFAULT_LEVELS, pairs=None, dyad_windows="period"
                           ) -> tuple[pd.DataFrame, pd.DataFrame, CoordinationReport]     # (pair, pair_phase, report)
def compute_cross_correlation(signals, *, levels=DEFAULT_LEVELS, pairs=None, dyad_windows="period"
                              ) -> tuple[pd.DataFrame, CoordinationReport]
def compute_vector_coding(signals, *, levels=DEFAULT_LEVELS, pairs=None
                          ) -> tuple[pd.DataFrame, pd.DataFrame, CoordinationReport]
def compute_coherence(signals, *, levels=DEFAULT_LEVELS, pairs=None) -> tuple[pd.DataFrame, CoordinationReport]
def compute_spectral(signals) -> tuple[pd.DataFrame, CoordinationReport]
def compute_cluster_phase(signals) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CoordinationReport]
    # (cluster_team, cluster_player, team_sync, report)
def compute_relative_stretch(signals) -> tuple[pd.DataFrame, CoordinationReport]
def compute_team_coordination(frames, *, windows=None, actions=None, params=None, goal_map=None, links=None,
                              levels=DEFAULT_LEVELS, dyad_windows="period") -> CoordinationResult
@dataclass(frozen=True)
class CoordinationResult:
    windows: pd.DataFrame
    pair: pd.DataFrame
    pair_phase: pd.DataFrame
    spectral: pd.DataFrame
    cluster_team: pd.DataFrame
    cluster_player: pd.DataFrame
    team_sync: pd.DataFrame
    rsi: pd.DataFrame
    report: CoordinationReport
def compute_coordination_series(signals, *, kind: Literal["relative_phase", "coupling_angle", "cluster_amplitude"],
                                pairs=None) -> pd.DataFrame
```

Per-method rules. Each is the spec section named, plus the concretisation where named.
- **Common.** For every (pair or team, window):
  - The window's valid samples are those inside `[start, end)`, valid in every series used, and inside a segment.
    Window sums come from prefix sums (`np.cumsum` over indicator and value arrays, one pass per series), so all
    windows cost O(N + W) per series (§7.6, R5).
  - Degradation tokens are checked in the precedence of Task 14.
  - `insufficient_detection` fires when either side's mean `observed_fraction` over the window is below
    `min_observed_fraction[family]`.
  - `no_possession_role` fires for an `attacking_defending` pair on a non-possession window.
  - Dyads run on `window_kind == "period"` only unless `dyad_windows == "all"` (D17).
  - C13: dyad rows carry `<NA>` in every `coord_vc_*`/`coord_coh_*` column and in their `_source` columns.
- **Relative phase** (§7.8.1, C6, C24):
  - `z = z_a · conj(z_b)`, times −1 when exactly one signal is positional and the window's reference team is B.
  - `(mean, R, sd) = circular_summary(Σz, n)`; the 12 histogram fractions and `pct_near_in_phase` come from
    prefix-summed bin and near-in-phase indicators.
  - `phase_valid_fraction_a/_b` is the share of the window's valid samples with positive instantaneous frequency,
    per side.
  - `n < 3` → `too_short`.
  - `COORD_PAIR_PHASE` rows: the same statistics over each `phase_assignment(m, n_phases)` subdivision of the
    window's **grid** samples (time thirds; C5), valid samples only, each subdivision also needing ≥ 3.
- **Cross-correlation** (§7.8.2):
  - Slices are the window ∩ segment runs of both-valid samples. Slices shorter than `min_slice_samples(L)` are
    skipped; none left → `too_short`.
  - Each slice goes through `lagged_pearson`, with the C24 sign.
  - The slices are pooled with `fisher_pool`, then summarised with `xcorr_summary`. An all-NaN r from a constant
    side → `degenerate_constant`.
- **Vector coding** (§7.8.3):
  - Differences are taken between consecutive grid samples that are both in the window and in the **same**
    segment, with the C24 signs.
  - Stationary samples are dropped using `vc_epsilon[signal]` and counted into `coord_vc_n_stationary` and the
    report.
  - Classification: `classify(coupling_angle_deg)`. The four fractions are over the non-stationary samples. The mean
    angle and its variability come from the circular summary of the angles.
  - Fewer than 3 non-stationary samples → `too_short`; a unit mismatch → `not_commensurate`.
  - `COORD_PAIR_PHASE` rows use the same statistics per subdivision.
- **Coherence** (§7.8.5):
  - `nperseg = round(welch_segment_s · fs)`. Slices of at least `nperseg` samples go through `welch_spectra`, then
    `pooled_coherence` over `[band_low_cpm, band_high_cpm]`.
  - `K < 4` → `too_short`.
- **Spectral** (§7.8.4):
  - One row per (team, `TEAM_SIGNALS` signal, window) plus one possession row (`team_id <NA>`, `signal="possession"`).
    The possession series is 1 when team A possesses, 0 when team B possesses, and excluded where NA.
  - Slices of at least `min_spectral_samples(fs, band_low_cpm)` go through `median_frequency_cpm` and are pooled by
    duration.
  - No possession windows in the call → `no_possession_role` on the possession row.
- **Cluster** (§7.8.6):
  - Per team and axis, over players' player-level phasors (the goalkeeper per `include_goalkeeper["cluster"]`,
    observed samples only): `cluster_phase` once per period, then `window_cluster_stats` per window.
  - `rho_group_sampen = sampen(rho_group_i)`. Player rows: `phi_mean_deg`, `rho_k`, `phi_sd_deg =
    degrees(sqrt(-2 ln rho_k))`, `phi_sampen = sampen(wrapped φ_k series)`, `on_pitch_s`.
  - No usable sample → `insufficient_players`; SampEn A or B = 0 → `entropy_undefined` (on the sampen column's
    group, i.e. the row source when every other statistic is scored).
- **Team sync** (§7.8.6, **C20**):
  - The two teams' **segment-level** instantaneous synchrony: ρ_group,i computed with φ̄_k over the continuous
    segment, i.e. the `cluster_amplitude` series of `compute_coordination_series`.
  - The window's samples where both are usable give `coord_team_sync_pearson_r` and `cross_sampen`.
  - C20 is required by §7.9: the team-sync surrogate "circularly shifts one team's ρ_group,i within each continuous
    segment", which presupposes a segment-level series.
- **RSI** (§7.8.7):
  - SI_A − SI_B per axis from `stretch_x`/`stretch_y`. Outputs: mean; fraction > 0; switch rate per minute (sign
    changes between consecutive valid samples in the same segment, divided by the valid minutes).
  - The bimodality coefficient uses `scipy.stats.skew(bias=False)` and `scipy.stats.kurtosis(fisher=True,
    bias=False)` in Pfister's formula.
  - `n < 4` → `too_short`; zero variance → `degenerate_constant`.
- **Surrogates** (§7.9, R1, C11):
  - Draws: `surrogate_rng(params.surrogate_seed, (game, period, segment_index, *pair_or_team_descriptor, family))`
    per segment. The shift threshold is `τ = round(min_shift_s[signal] · fs)`, where the signal is signal B, the
    player for cluster, or `"cluster_amplitude"` for team sync.
  - Estimators: phasor sums through `shifted_phasor_sums`; near-in-phase through `shifted_near_in_phase_counts`;
    whole-segment cross-correlation through `shifted_lagged_pearson`, and a direct roll otherwise; vector coding and
    coherence by direct recomputation on the rolled series; cluster by independently rolled player phasors; team
    sync by rolling team B's `cluster_amplitude`.
  - `surrogate_method == "iaaft"` replaces rolling with `iaaft` on the **signal** of each segment, recomputing the
    phase through `analytic_phase` (estimator identity). Non-convergence is counted in `iaaft_nonconverged`, **and**
    carried per row (A1).
  - The surrogate source is the first rule that applies, in this order:
    1. `disabled` when `n_surrogates == 0`;
    2. `not_scored` when the observed metric is not scored;
    3. `segment_too_short` (time-shift only) when any contributing segment (or player run, for cluster) has
       `shift_bounds(...) is None`;
    4. `computed_nonconverged` (IAAFT only, A1) when any IAAFT draw contributing to the row returned
       `converged=False`;
    5. otherwise `computed`.
  - The triple comes from `surrogate_triple`.
- **Orchestrator:**
  - `windows=None` means `period_windows(frames)` concatenated with `possession_windows_from_actions(actions,
    frames, links=links, n_phases=params.n_phases)` when `actions` is given, else
    `possession_windows_from_frames(frames, n_phases=params.n_phases, params=params)` (§7.2, C25).
  - `params=None` means `CoordinationParams.for_provider(<the single source_provider>)`.
  - It builds the signals once and runs all seven families.
  - It outer-joins the four pair tables on `COORD_PAIR_KEYS`. Keys are already canonical, and `align_join_keys` is
    applied anyway. It asserts the coverage and provenance columns agree across families (a violation is a
    `RuntimeError`, i.e. a bug).
  - It concatenates the two `COORD_PAIR_PHASE` tables on their keys, and merges the reports.
  - Warning: the family computes emit `CoordinationCoverageWarning` (`stacklevel=2`) when the dropped share exceeds
    `coverage_warn_fraction`. The orchestrator suppresses the families' warnings and emits one.
- **`compute_coordination_series`.** A long table:
  - `relative_phase` / `coupling_angle`: `game_id, period_id, segment_id, time_s, kind, level, signal_a, signal_b,
    axis, team_a_id, team_b_id, player_a_id, player_b_id, value` (degrees; stationary coupling samples omitted).
  - `cluster_amplitude`: `game_id, period_id, segment_id, time_s, kind, team_id, axis, value` (C20).
  - Its reference frame is team A's, since there are no windows. It has no contract: it is a raw primitive (§7.12).

- [ ] **Step 1: Write the failing tests.**
  - `test_compute.py` (synthetic fixture with known coupling):
    - `test_team_centroid_relative_phase_recovers_planted_offset`: the longitudinal team_team `centroid_x` row on the
      period window has mean ≈ +40° (within 3°) and R > 0.9.
    - `test_positive_lag_means_a_leads`: team B's centroid delayed by 2 s → `coord_xc_lag_s ≈ +2.0` on the
      `spread` pair.
    - `test_vector_coding_fractions_sum_to_one`.
    - `test_vector_coding_differences_never_span_a_split`: a planted jump across a stoppage split produces no
      coupling sample.
    - `test_vector_coding_min_samples_per_subdivision_both_sides`.
    - `test_coherence_near_one_for_coupled_centroids`.
    - `test_spectral_row_per_team_signal_and_possession`.
    - `test_cluster_rho_high_for_synchronised_team`.
    - `test_team_sync_pearson_positive_for_common_drive`.
    - `test_rsi_sign_and_bimodality_on_alternating_stretch`.
    - `test_dyads_period_only_by_default_and_all_when_requested` (D17).
    - `test_dyad_rows_have_null_vc_and_coh_by_level` (C13).
    - `test_pair_phase_subdivisions_use_time_thirds` (C5).
    - `test_surrogate_triple_present_and_percentile_separates_coupled` (non-vacuity: coupled R percentile ≥ 0.99;
      decoupled in [0.05, 0.95]).
    - `test_iaaft_mode_runs_and_counts_nonconvergence`. Checked from both sides:
      - With `iaaft_max_iter=1` on a strongly non-Gaussian series, at least one draw fails to converge. The affected
        rows carry `computed_nonconverged`, and the report counter equals the number of those draws.
      - With the default `iaaft_max_iter` on a Gaussian AR(1) series, every draw converges and the rows carry
        `computed` (A1).
    - `test_surrogates_disabled_when_n_surrogates_zero` (R1 `disabled`).
    - `test_surrogate_segment_too_short` (R1).
    - `test_orchestrator_tables_match_family_computes`.
    - `test_orchestrator_default_windows_period_plus_one_possession_source` (C25).
    - `test_single_warning_from_orchestrator_both_sides_of_threshold`.
    - `test_determinism_same_seed_same_output_across_calls_and_window_order`.
    - `test_orientation_signs_c24`: `_orientation_signs` covers all 4 kind combinations × 2 references.
  - `test_degradation.py`:
    - `test_every_source_token_reachable`: parametrised over `COORD_SOURCE_VALUES`, one planted input each. For
      example, `goal_end_unresolved` via frames whose keepers are removed and whose outfield means sit at 52.5, so
      the `GoalMap` cannot guess; `not_commensurate` via a `PairSpec("team_team", "convex_hull_area", "spread",
      "canonical")`.
    - `test_every_surrogate_token_reachable` (R1).
    - `test_report_conserves_windows_and_rows` (`conservation_errors() == []` for the full orchestrator run and every
      family).
    - `test_drop_reasons_attributed_by_precedence`.
  - `test_series.py`:
    - `test_relative_phase_series_matches_pair_statistics`: the circular mean of the series over the period window
      equals the pair row's `coord_rp_mean_deg`.
    - `test_coupling_angle_series_omits_stationary`.
    - `test_cluster_amplitude_is_segment_level` (C20).
    - `test_long_table_columns`.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement.** Loops run over (pair, window) and (team, window) index ranges on arrays; no DataFrame is
  filtered inside a loop (ADR-068). Output frames are assembled once per family from column arrays, then cast to the
  `*_COLUMNS` dtypes. Ids are canonical in pair keys and `restore_id_dtype`-restored for single-entity ids (C22).
- [ ] **Step 4: Run** → **PASS**, including `SILLY_KICKS_COORDINATION_FORCE_NUMPY=1`. Then ruff + format + pyright.

---

### Task 18: Public surface and every new-package gate (§9.3–§9.5)

**Files:**
- Modify:
  - `silly_kicks/coordination/__init__.py`
  - `silly_kicks/metric_contracts.py`
  - `tests/test_metric_contracts.py` (`_PKG`)
  - `silly_kicks/feature_glossary.py`
  - `NOTICE`
  - `tests/invariants/glossary_emitted_columns.py`
  - `tests/invariants/test_glossary_emitted_columns.py`
  - `tests/test_public_api_examples.py` (`_PUBLIC_MODULE_FILES`)
  - `tests/_scale_guarded.py`
  - `tests/test_scale_guards.py`
  - `docs/c4/architecture.dsl` and `architecture.html`
  - `docs/PRIVATE_CONSUMERS.md`
- Create:
  - `tests/coordination/test_public_surface.py`
  - `tests/coordination/test_import_allowlist.py`
  - `tests/coordination/test_purity.py`
  - `tests/coordination/test_id_dtype_invariance.py`
  - `tests/coordination/test_orientation_invariance.py`
  - `tests/coordination/test_liveness.py`
  - (R2) `tests/datasets/tracking/idsse_half/{generate_fixture.py, README.md, frames.parquet, actions.parquet}`

- [ ] **Step 1: The public surface.** `silly_kicks/coordination/__init__.py` exports exactly:
  - the §7.2 functions and classes: `CoordinationParams`, the 3 window builders, `build_coordination_signals`, the 7
    family computes, `compute_coordination_series`, `compute_team_coordination`, `CoordinationSignals`,
    `CoordinationResult`, `CoordinationReport`, `CoordinationCoverageWarning`;
  - `PairSpec`, `DEFAULT_PAIRS`, `DEFAULT_LEVELS`;
  - every `*_KEYS`, `*_METRIC_COLUMNS` and `*_COLUMNS` constant;
  - `COORD_SOURCE_VALUES`, `COORD_SURROGATE_SOURCE_VALUES`, `COORD_LEVELS`, `COORD_SIGNALS`,
    `COORD_WINDOW_COLUMNS`.

  `test_public_surface.py::test_all_is_exact` asserts set equality with a literal list, plus
  `test_every_public_name_importable`.
- [ ] **Step 2: The import allowlist** (`tests/coordination/test_import_allowlist.py`). It mirrors
  `tests/restdefense/test_import_allowlist.py`, with AST scans using `rglob`:
  - `test_kernels_import_only_numpy_scipy_stdlib_numba`: every module under `coordination/_kernels/` imports only
    `numpy`, `scipy`, the standard library and `numba`.
  - `test_coordination_private_imports_are_allowlisted`:
    - `_PRIVATE_IMPORT_ALLOWLIST` is the exact set of (module stem, private module) pairs, each with a reason
      comment (C10): `tracking._provider_visibility` in `_signals` and `_windows`; `tracking._geometry` in
      `_signals`; `tracking._collective` array kernels; `tracking.preprocess._butterworth` (`winter_correction`,
      `butterworth_min_length`).
    - The public tracking seams are imported from `silly_kicks.tracking`.
  - `test_nothing_imports_coordination`: no module in `silly_kicks` outside `coordination/` imports it.
  - Planted-violation meta-tests, one per direction: a synthetic source string with a forbidden import is detected.
- [ ] **Step 3: `metric_contracts`.**
  - Add the 7 hard-coded families to `METRIC_CONTRACTS`: `keys` / `metric_columns` / `columns` tuples and
    `column_types` = the `*_COLUMNS` dict.
  - Add 7 `_PKG` rows in `tests/test_metric_contracts.py`, e.g.
    `"coordination_pair": ("silly_kicks.coordination", "COORD_PAIR_KEYS", "COORD_PAIR_METRIC_COLUMNS", "COORD_PAIR_COLUMNS")`.
  - Before the rows are added, the re-keyed completeness test (Task 6) fails on the new package. That is the
    intended detection; the rows make it pass.
- [ ] **Step 4: Invariants.**
  - `test_purity.py` checks every public compute and builder against a deep copy of its inputs (`frames`, `actions`,
    `windows`, `links`).
  - `test_id_dtype_invariance.py`: int, string and `category` team/player/game ids give identical metric columns,
    with key columns equal after `canonical_id_series`.
  - `test_orientation_invariance.py`:
    - Mirror: reflect x → 105 − x and y → 68 − y, swap `team_attacking_direction`, rebuild → every metric column
      equal within `rtol=1e-9, atol=1e-9`, **including** `coord_vc_mean_angle_deg` (compared circularly).
    - Identity: relabel team ids → outputs identical up to the relabel.
    - Non-vacuity: with orientation disabled by monkeypatching the transform to identity, the mirror test's
      `coord_vc_mean_angle_deg` differs by 180°.
- [ ] **Step 5 (R2): The real long fixture generator.** `tests/datasets/tracking/idsse_half/generate_fixture.py`:
  - Loads DFL match `J03WMX` (the Sportec Open DFL Dataset, Bassek et al. 2025, CC BY 4.0; the same match and licence
    as `tests/datasets/elastic_sync/j03wmx_slice`).
  - Uses the repo's own `pining` loader (`scripts/_loader_pining.load_match(ref, events_only=False)` for provider
    `idsse`; token from the environment), producing canonical `TRACKING_FRAMES_COLUMNS` frames and SPADL actions.
  - Cuts period 1 to `time_seconds < 1200`, keeps every frame whose index is even (12.5 Hz), keeps only the columns
    TF-58 reads, and writes `frames.parquet` and `actions.parquet` with pinned parquet settings (engine `pyarrow`,
    `compression="zstd"`, `index=False`, sorted by `frame_id, player_id`).
  - Records row counts and file sizes in `README.md`. The README carries the CC BY 4.0 attribution block copied from
    the j03wmx README.
  - `test_idsse_half_generator_reproduces_fixture` (marked `e2e`, since it needs the token) proves byte-for-byte
    reproduction (ADR-056).
  - If the fixture exceeds 6 MB, stop and report the size to the owner before committing to it.
- [ ] **Step 6 (R2): Fixture preconditions.** These run in CI (`test_liveness.py::test_idsse_half_preconditions`):
  - the duration is ≥ 1,100 s;
  - `ball_state` has at least one dead interval > 25 s;
  - the actions cover the span, with at least one goal and at least 5 restarts;
  - both teams have 10 outfield players for ≥ 90% of samples.
- [ ] **Step 7: Liveness** (`test_liveness.py`, ADR-032).
  - For each committed provider fixture (the Task 1 list plus the R2 `idsse_half`), run `compute_team_coordination`
    with `n_surrogates=19` (liveness needs values, not the K = 199 chance resolution).
  - Assert that every `*_METRIC_COLUMNS` column of every table is non-NaN in at least one row **and** non-constant
    across rows, over the union of fixtures.
  - Each fixture gets a precondition test asserting it satisfies its method minima (§7.8). The R2 fixture carries
    spectral and coherence.
  - Heavy combinations are `@pytest.mark.slow` (ADR-023): the full surrogate run on `idsse_half`.
  - *If R2 is declined:* spectral and coherence liveness run on `make_coordination_match(seconds=2700)`, and
    `test_committed_real_fixtures_too_short_for_spectral_and_coherence` proves the real fixtures are below both
    minima.
- [ ] **Step 8: Structural guards** (§9.4, R8).
  - `SCALE_GUARDED` gains `silly_kicks.coordination._signals.build_coordination_signals` and every other
    `group_rows` caller the AST discovery finds in `coordination/`.
  - Guards in `tests/test_scale_guards.py`:
    - `test_build_coordination_signals_is_subquadratic` (`rows_scanned_counter` over 8/16/32 synthetic games);
    - `test_relative_phase_is_linear_in_pairs` (work counted by `call_counter` on `circular_summary`, over 8/32/128
      caller `PairSpec`s).
  - `test_regressed_rescan_goes_quadratic` monkeypatches a rescan into the per-period loop and asserts the growth
    guard fails. That is the proof §9.4 asks for.
- [ ] **Step 9: Glossary, NOTICE, emitted columns** (ADR-048, ADR-005).
  - `Unit` gains `"cycles/min"` (the §9.4 ADR-048 amendment) and, per **R5**, `"switches/min"`.
  - One `FeatureColumn` per metric column across the 7 tables (unique names; the unit per C22 and R5):

    | Unit | Columns |
    |---|---|
    | `degrees` | `*_deg` |
    | `ratio` | R, `pct`/`fraction`/`percentile`, coherence, ρ, observed fraction |
    | `seconds` | `lag_s`, `duration_s`, `on_pitch_s` |
    | `count` | `n_*` |
    | `cycles/min` | `median_freq_cpm`, `peak_freq_cpm` |
    | `switches/min` | `rsi_switch_rate_per_min` |
    | `metres` | `rsi_mean_m` |
    | `dimensionless` | r, SampEn, bimodality coefficient; `*_excess` takes its metric's unit |

  - Each definition states the published reference range where one exists (§5 goal 4), e.g. "Moura 2016 reported max
    |r| 0.41 ± 0.09 (defending) …".
  - `emitting_module` is the family's home, e.g. `silly_kicks.coordination._compute`.
  - `attribution` is a TF-58 token; add the constants beside the existing `_A_*` block:
    - `_A_TF58_BOURBOUSSON = "Bourbousson, Seve & McGarry (2010)"`
    - `_A_TF58_MOURA_2016 = "Moura et al. (2016)"`
    - `_A_TF58_MOURA_2013 = "Moura et al. (2013)"`
    - `_A_TF58_FOLGADO = "Folgado et al. (2014)"`
    - `_A_TF58_RICHARDSON = "Richardson et al. (2012)"`
    - `_A_TF58_DUARTE = "Duarte et al. (2013)"`
    - `_A_TF58_RICHMAN = "Richman & Moorman (2000)"`
    - `_A_TF58_WELCH = "Welch (1967)"`
    - `_A_TF58_PFISTER = "Pfister et al. (2013)"`
  - Coverage columns have attribution `None`.
  - `NOTICE` gains a TF-58 methodology paragraph citing every §2 source (Appendix A), containing each token
    verbatim. It also cites Shewchuk, J. R. (1997), "Adaptive Precision Floating-Point Arithmetic and Fast Robust
    Geometric Predicates", *Discrete & Computational Geometry* 18(3):305–363, for the exact orientation predicate in
    `tracking/_collective.py` (A2, ADR-005).
  - `tests/invariants/glossary_emitted_columns.py` gains `_coordination_columns()`: it runs
    `compute_team_coordination` on the synthetic fixture and returns every `*_METRIC_COLUMNS` value, and is wired
    into `emitted_columns()`.
  - `test_each_leg_is_non_vacuous` gains `assert "coord_rp_resultant_length" in E._coordination_columns()`.
- [ ] **Step 10: Public API examples.** Add the modules that define the public surface to `_PUBLIC_MODULE_FILES`:
  `coordination/_config.py`, `_windows.py`, `_catalog.py`, `_signals.py`, `_compute.py`, `_series.py`, `_report.py`,
  with a comment in the restdefense style. Every public function, class and method carries an Examples section:
  - a doctest where it runs without frames;
  - an RST literal block otherwise.
- [ ] **Step 11: The C4 model.**
  - Add `coordination = container "silly_kicks.coordination" "<≤ 200 chars: TF-58 team-coordination dynamics:
    relative phase, cross-correlation, vector coding, spectral, cluster phase; surrogate baselines. ADR-NNN.>"`.
  - Add the relationship `coordination -> tracking "Consumes PUBLIC tracking seams (collective kernel, preprocess,
    GoalMap, carrier) + two allowlisted privates"`.
  - Update the glossary count sentence to `len(FEATURE_GLOSSARY)`.
  - Render per the owner's C4 rule: `structurizr.war export` → `c4_assemble.py --inject-wrap-width` →
    `plantuml.jar -graphvizdot "C:/Users/Karsten/.claude/tools/graphviz/dot.exe" -tsvg` → `c4_assemble.py
    --svg-dir`. A clean assemble proves `dot` ran.
  - Run `tests/test_c4_dsl_description_cap.py`, `tests/test_c4_feature_column_count.py` and
    `tests/test_c4_aggregator_count.py`.
- [ ] **Step 12: `docs/PRIVATE_CONSUMERS.md`** (C10). Add `silly_kicks.coordination` as a consumer:
  - the `_provider_visibility` row gets `dead_ball_observed`, `_DEAD_BALL_OBSERVED_PROVIDERS` and
    `_DETECTION_AWARE_PROVIDERS`;
  - the `_geometry` row gets the two array twins;
  - the `_frame_index` row gets `coordination/`;
  - a new row covers `tracking/_collective.py` and `tracking/preprocess/_butterworth.py` private helpers used by
    coordination.
- [ ] **Step 13: Run the registries.**
  - Run `tests/invariants/test_public_id_scalar_registry.py`. If its enumeration picks up a new public function,
    register it per the file's existing idiom.
  - Run `tests/test_add_star_purity.py`: no `add_*` was added, so it is unchanged.
  - Run every file created or touched in Steps 1–12, then `tests/coordination tests/tracking tests/invariants
    tests/test_metric_contracts.py tests/test_feature_glossary_coverage.py
    tests/test_feature_glossary_notice_linkage.py tests/test_scale_guards.py tests/test_public_api_examples.py
    tests/test_ci_shard_wiring.py -m "not e2e"` → **PASS**.
  - Run `--doctest-modules silly_kicks/coordination` → PASS. It covers private-module examples too, which the CI
    doctest job's `--ignore-glob` skips.
  - Then ruff + format + pyright.

---

## Drivers (Tasks 19–23)

Shared discipline for D1, D2 and D3 (§8.3):
- Load and resume:
  - `scripts/_driver.for_each(refs, key=lambda ref: ref.key, load=_load, work=..., shard_root=dest / "shards",
    token_inputs={...}, label="match")`; resume happens before load (ADR-052).
  - `load` is `pining_source(...)[1]`, i.e. `load_match(ref, events_only=False)` (rules A–D stay green).
  - `assert_conservation` and `_require_injective` are applied.
- Provenance and preflight:
  - `require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)`.
  - Every driver has a module-level `input_contract()` returning `declare_inputs(driver="<stem>", ...)`.
  - Preflight `scripts._corpus_visibility.validate_corpus_visibility` (ADR-069 Layer 2).
- Outputs: aggregate-only, labelled per ADR-038, written with `write_table_atomically` / an atomic JSON write.
- Corpus scope:
  - disjoint worker slices via `--match-ids-json` and `providers_for_slice`;
  - per-stage timing counters in the manifest (R8);
  - the full corpus, never shrunk.
- CLI flags: `--out`, `--token`, `--max-matches`, `--cache-dir`, `--match-ids-json`, `--allow-dirty`, `--list-matches`
  (the `scripts/validate_territorial_defense.py` shape) plus `--providers` (default
  `skillcorner,gradientsports,idsse`: the ~980-match TF-58 corpus, 909 + 64 + 7).

### Task 19: Shared driver modules

**Files:**
- Create:
  - `scripts/_coordination_corpus.py`
  - `scripts/_coordination_occlusion.py`
  - `scripts/_coordination_params_codegen.py`
  - `scripts/_coordination_thresholds.py`
  - `scripts/_coordination_hypotheses.py`
- Test: `tests/scripts/test_coordination_driver_modules.py`

**Interfaces — produces:**

```python
# _coordination_corpus.py
TF58_PROVIDERS = ("skillcorner", "gradientsports", "idsse")
def add_common_args(parser: argparse.ArgumentParser) -> None
def corpus_source(args) -> tuple[list, Callable]             # pining_source(providers_for_slice(...), match_ids=...,
                                                             # max_per_provider=args.max_matches, cache_dir=..., token=...)
def match_tables(loaded, params: CoordinationParams, *, n_surrogates: int,
                 stoppage_evidence: str = "auto", detection: str = "auto",
                 variants: Mapping[str, CoordinationParams] | None = None) -> pd.DataFrame
    # Builds windows (period + possession from events) and signals ONCE; runs the 7 families per params variant
    # (post-preparation variants reuse the prepared signals via dataclasses.replace(signals, params=v) -- §8.4);
    # returns one long frame: provider, match_id, variant, table, then the table's columns as a JSON-able wide row.
class StageTimer: ...                                        # context-manager stage timer -> manifest counters (R8)

# _coordination_occlusion.py
def fov_mask(frames: pd.DataFrame, *, width_m: float) -> np.ndarray
    # per row: x inside [c - W/2, c + W/2], c = clip(ball x of the frame, W/2, 105 - W/2); ball rows True
def simulate_broadcast_occlusion(frames: pd.DataFrame, *, width_m: float) -> pd.DataFrame
    # visibility := fov_mask; masked player positions replaced by linear interpolation between that player's
    # detections (edges held) -- an approximation of SkillCorner's own extrapolation, labelled as one (§8.2)
def detection_rates(frames: pd.DataFrame, mask: np.ndarray) -> tuple[float, float]   # (outfield, goalkeeper)
def calibrate_width(frames_list: Sequence[pd.DataFrame], *, target_outfield: float = 0.666,
                    tol: float = 0.002) -> float                                     # bisection on W in [5, 105]

# _coordination_params_codegen.py
GENERATED_PATH = pathlib.Path("silly_kicks/coordination/_provider_params_generated.py")
INTERIM_BASE: Mapping[str, object]
    # the R4 table (complete maps for min_observed_fraction / vc_epsilon / min_shift_s), each value commented with its
    # rule -- the ONLY place the interim numbers are written (A3)
def render_generated_params(derivation: Mapping | None, calibration: Mapping | None) -> str
    # Deterministic: providers sorted, keys sorted, floats via repr(float(v)). Emits three names:
    #   BASE_SOURCE = "interim" | "derivation"
    #   BASE_COORDINATION_PARAMS = INTERIM_BASE when derivation is None, else derivation["pooled"] (A3)
    #   PROVIDER_COORDINATION_PARAMS = {} when derivation is None, else derivation["providers"]
    # calibration's MOVED selections are applied to both the pooled base and every provider (same multiplier/offset);
    # (None, None) renders exactly the commit-1 file
def write_generated_params(text: str) -> None                                       # atomic replace

# _coordination_thresholds.py -- pre-registered (§8.5), referenced never inlined
H1_P = 0.01
H1_LONGITUDINAL_MEAN_WITHIN_DEG = 30.0
H2_POSITIVE_SHARE = 0.70
H2_MEDIAN_ABS_LAG_S = 1.0
H3_P = 0.05
H4_BELOW_CEIL_SHARE = 0.95
H4_CEIL_CPM = 1.0
H4_P = 0.01
H5_P = 0.01
H5_TOST_MARGIN = 0.05
H5_TOST_ALPHA = 0.05
H7_BC_THRESHOLD = 5.0 / 9.0
H7_BIMODAL_SHARE = 0.50
H7_SWITCH_WINDOW_S = 10.0
H7_SURROGATE_PERCENTILE = 95.0
STOPPAGE_LEG_MIN_S = 25.0
GATED_HYPOTHESES = ("H1", "H2", "H3", "H4", "H5", "H7")    # H6 is descriptive

# _coordination_hypotheses.py -- pure reducers over the per-match tables; each returns {"pass": bool | None, ...stats}
def h1_centroid_phase_stability(pair: pd.DataFrame) -> dict   # paired sign test R_x > R_y (binomtest, one-sided) + pooled circ mean
def h2_spread_xcorr(pair: pd.DataFrame) -> dict
def h3_early_third_by_terminal(pair_phase: pd.DataFrame, windows: pd.DataFrame) -> dict  # mannwhitneyu one-sided x2
def h4_median_frequency(spectral: pd.DataFrame) -> dict        # share < 1 cpm + wilcoxon signed-rank first > second
def h5_rho_group(cluster_team: pd.DataFrame, windows: pd.DataFrame) -> dict   # sign test x > y + paired TOST in/out possession
def h6_dyad_near_in_phase(pair: pd.DataFrame) -> dict          # quantiles per axis; "pass": None
def h7_rsi(rsi: pd.DataFrame, rsi_switch_times: pd.DataFrame, possession_changes: pd.DataFrame, *, seed: int) -> dict
def evaluate_hypotheses(tables: Mapping[str, pd.DataFrame], *, seed: int) -> dict[str, dict]
def gated_pass(results: Mapping[str, dict]) -> bool            # every GATED_HYPOTHESES entry passes
def boundary_f1_by_gap(frames, actions, gaps_s: Sequence[float]) -> dict[float, float]
    # frames-only possession spells (possession_windows_from_frames with possession_gap_s = g) vs event possessions,
    # both mapped to a per-frame possession id; silly_kicks.spadl.utils.boundary_metrics -> f1 (the TF-52 idiom)
```

The statistics are the scipy primitives named in §8.5:

| Test | scipy call |
|---|---|
| sign test | `scipy.stats.binomtest(k, n, 0.5, alternative="greater")` |
| Wilcoxon rank-sum | `scipy.stats.mannwhitneyu(..., alternative="greater")` |
| Wilcoxon signed-rank | `scipy.stats.wilcoxon(..., alternative="greater")` |
| TOST | two `scipy.stats.ttest_1samp` calls on the paired differences (`alternative="greater"` against `-margin`, `"less"` against `+margin`); equivalent iff both p < α |

- [ ] **Step 1: Write the failing tests.**
  - Codegen:
    - `test_codegen_empty_reproduces_commit_1_file`: `render_generated_params(None, None)` equals the committed
      `_provider_params_generated.py` byte-for-byte, including `BASE_SOURCE = "interim"` and `INTERIM_BASE` as the
      base.
    - `test_codegen_deterministic_and_sorted`.
    - `test_codegen_applies_only_moved_selections`, to the pooled base **and** to every provider (A3).
    - `test_codegen_with_derivation_writes_pooled_base` (A3): a planted derivation with a `pooled` block renders
      `BASE_SOURCE = "derivation"` and `BASE_COORDINATION_PARAMS == derivation["pooled"]`.
    - `test_interim_base_matches_R4_table`: `INTERIM_BASE` equals the R4 table value for value.
  - Occlusion:
    - `test_fov_mask_clamps_to_pitch` (ball at x = 2 with W = 40 gives the window [0, 40]).
    - `test_calibrate_width_hits_target_rate` (synthetic 22-player match: within `tol`).
    - `test_occlusion_interpolates_masked_positions_and_flags_visibility`.
  - Thresholds: `test_thresholds_match_spec_table`, one assert per constant.
  - Hypothesis reducers, one pair of tests **per hypothesis**, both sides of its threshold (§8.5, "every band tested
    from both sides"):
    - H1: R_x > R_y in 20/20 halves passes; 12/20 fails. Circular mean 25° passes; 35° fails.
    - H2: positive share 0.75 passes; 0.65 fails. Median |lag| 0.9 s passes; 1.1 s fails.
    - H3, H4, H5: planted separations pass, and planted nulls fail. H5's TOST passes at a true difference of 0.01 and
      fails at 0.08.
    - H6 always returns `"pass": None`.
    - H7: BC share 0.6 passes; 0.4 fails. The switch share above the planted surrogate passes; below it fails.
  - `test_gated_pass_ignores_h6`.
  - `test_boundary_f1_by_gap_prefers_true_gap`: a planted frames/actions pair where gap 1.0 s yields the highest F1.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → **PASS**. Also run `tests/scripts/test_corpus_driver_resilience.py`: rules A–D scan the new
  private modules, and none of them loads. Then ruff + format + pyright.

---

### Task 20: D1 `scripts/derive_coordination_params.py` (Tier B, §8.2)

**Files:**
- Create: `scripts/derive_coordination_params.py`
- Modify:
  - `tests/scripts/test_provenance_wiring.py` (`ARTIFACT_DRIVERS` += `"derive_coordination_params"`, with a comment in
    the house style)
  - `tests/scripts/test_input_contracts.py` (`_DECLARING` += the stem; detector widening, C3)
- Test: `tests/scripts/test_derive_coordination_params.py`

**CLI:** `--pass {a,b,occlusion,reduce}` plus the common arguments. Each of `a`, `b` and `occlusion` is its own
shardable `for_each` pass (ADR-052 "expensive passes are their own drivers or passes"). `reduce` reads the three shard
generations, writes `docs/research/tf58_team_coordination/derivation.json`, and regenerates
`_provider_params_generated.py` through `render_generated_params(derivation, None)`.

**Pure reducers (module-level, unit-tested):**

```python
def residual_cutoff_for_run(values: np.ndarray, fs: float) -> float    # residual_analysis_cutoff on the
                                                                       # 0.1..5.0 Hz / 0.05 grid (clipped, C18)
def weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> float
    # np.quantile(values, q, weights=weights, method="inverted_cdf") -- the one quantile definition every reducer uses
def unit_weights(provider: np.ndarray, match: np.ndarray, *, provider_neutral: bool) -> np.ndarray
    # w = 1 / U_m within each match (each match counts once), times 1 / M_p within each provider when
    # provider_neutral (each provider counts once); normalised to sum 1. Per-provider values use
    # provider_neutral=False on that provider's units; the pooled base uses provider_neutral=True (A3, TF58-PLAN-02).
def provider_cutoff(cutoffs: np.ndarray, weights: np.ndarray) -> float           # weighted median over runs
def vc_epsilon_for_signal(raw_resampled: np.ndarray, filtered: np.ndarray) -> float
    # 1.4826 * MAD(diff(raw - filtered)) per match; combined across matches by weighted median
def first_acf_zero_crossing_s(x: np.ndarray, fs: float) -> float       # first lag with ACF <= 0 (linear interp)
def min_shift_for_signal(zero_crossings_s: np.ndarray, weights: np.ndarray) -> float   # weighted 95th pct, match-halves
def band_from_median_frequencies(values_cpm: np.ndarray, weights: np.ndarray) -> tuple[float, float]
    # weighted 5th and 95th percentiles
def welch_segment_rule(band_low_cpm: float, band_high_cpm: float, half_s: float = 2700.0) -> tuple[float, bool]
    # -> (segment_s, resolution_won); the longest segment with resolution <= (band width)/4 AND >= 8 segments per
    # half at 50% overlap; if both cannot hold, resolution wins and the flag records the shortfall.
    # Rounded UP to the next 10 s.
def possession_gap_argmax(f1_by_gap: Mapping[float, float]) -> float  # grid {0.2, 0.4, ..., 3.0}; ties -> smaller gap
    # input: the WEIGHTED mean F1 curve (per-match curves, unit_weights with each match as one unit)
def min_observed_fraction_rule(err_by_bin: Mapping[float, float], between_match_sd: float) -> float
    # the smallest observed-fraction bin whose median |error| <= 0.5 * SD (§8.2)
def max_detection_gap_rule(rmse_by_gap_s: Mapping[float, float], noise_rms: float) -> float
    # the longest gap with RMSE <= 2 * noise RMS (§8.2)
def provider_bootstrap_se(units: pd.DataFrame, reducer: Callable[[pd.DataFrame], float], *, seed_key: tuple,
                          n_boot: int = 1000) -> float
    # TF58-PLAN-03: resample the provider's MATCHES with replacement (seeded via
    # SeedSequence(entropy=<input_contract digest as int>, spawn_key=key_words(seed_key))), re-apply the same reducer
    # with per-provider unit_weights on each resample, and return the std (ddof=1) of the 1000 values -- the match-level
    # uncertainty of that provider's estimate. n_boot = 1000, the conventional bootstrap count (>= Efron & Tibshirani's
    # 200 for standard errors).
def thin_provider_flags(per_provider: Mapping[str, float], se: Mapping[str, float]) -> dict[str, str]
    # Per quantity: provider p -> "flagged" when se[p] > SD(ddof=1) of the per-provider values (its own uncertainty
    # exceeds the between-provider spread, so it is too noisy to count as one draw); else "ok"; every provider
    # "not_assessable" when fewer than 2 providers carry the quantity. REPORTS only -- never drops or re-weights.
def input_contract() -> dict   # declare_inputs(driver="derive_coordination_params", grid=..., percentiles=...,
                               # geometry_version=GEOMETRY_VERSION, params_defaults=asdict(CoordinationParams()))
```

**The passes:**
- **Pass A** (raw positions, native rate): one shard row per (provider, match, player, axis, run) with the run's
  residual-analysis cutoff.
- **Pass B** (its `token_inputs` include the pass-A reduced cutoffs):
  - per signal: the MAD epsilon;
  - per signal and match-half: the ACF zero crossing;
  - per team-signal and half: the median frequency;
  - per event match: `boundary_f1_by_gap` over the gap grid.
- **Occlusion pass** (GS + IDSSE, fully observed):
  - Calibrate W globally with `calibrate_width` on a first sub-pass over the same shards. Its result, W, is a token
    input of the metric sub-pass.
  - Record the resulting goalkeeper rate against 19.6%.
  - Per match: full-observation metrics vs occluded metrics (`detection="detection_aware"`, C12), binned by observed
    fraction.
  - Per gap length: the RMSE of bridged positions.
- **Reduce:**
  - `derivation.json` holds:
    - the per-provider values (`providers`);
    - **the pooled, corpus-wide values (`pooled`, A3)**;
    - every intermediate distribution summary;
    - the width W and the goalkeeper-rate check;
    - `input_contract`, `run_commit`, `run_tree_dirty` and the stage timings.
  - **Intended population of the pooled base (TF58-PLAN-02): a provider not in the map.** The base default serves
    `CoordinationParams()` and every provider absent from `PROVIDER_COORDINATION_PARAMS`, which is by construction an
    unseen provider. Tier-B quantities are properties of a provider's *data*: noise floors, filter cutoffs, detection
    behaviour. So each **provider** is one draw from that population, and the base must be **provider-neutral**.
    A corpus-representative (match-weighted) base would inherit SkillCorner's signature, because 909 of the corpus's
    980 matches (92.8%) are SkillCorner. The plan's earlier "match-weighting stops SkillCorner dominating" rationale
    was wrong, and this bullet replaces it.
  - **Pooling:** `pooled` applies the **same** reducer functions to every provider's units (runs, match-halves, team
    signals, event matches) with `unit_weights(..., provider_neutral=True)`:
    - each provider carries equal total weight;
    - within a provider, each match carries equal weight;
    - within a match, each unit carries equal weight.

    Examples: `provider_cutoff`'s weighted median of all runs; `band_from_median_frequencies`' weighted percentiles
    of all team-halves; `possession_gap_argmax` over the weighted mean F1 curve; the occlusion-derived rules over the
    GS and IDSSE error curves at equal provider weight.
  - **Per-provider values** use the same reducers with `provider_neutral=False` on that provider's own units. One
    weighting scheme (`unit_weights`) and one quantile definition (`weighted_quantile`) serve both, so the two are
    directly comparable.
  - The derivation records each provider's total weight and match count beside `pooled`. It also records, as a
    diagnostic, the corpus-representative (match-weighted) value of each quantity, so the size of the SkillCorner
    effect is visible.
  - **Thin providers (TF58-PLAN-03, owner-approved 2026-09-26).**
    - *Equal provider weighting is deliberate.* The base describes an unseen provider, so each provider is one draw,
      whatever its size.
    - *There is no minimum-match floor.* Any floor would be an arbitrary constant, and it would silently remove a
      provider class from the population.
    - *Precision is measured instead.* For every provider and every Tier-B quantity (each scalar, and each
      per-signal/per-family map entry), D1 computes `provider_bootstrap_se`. `thin_provider_flags` then flags a provider
      whose estimate is noisier than the between-provider spread.
    - *Where it is recorded.* `derivation.json` carries a `thin_providers` block: per quantity, each provider's value,
      its SE, the between-provider SD and its flag. The reduce step prints every flag.
    - *Flags go to the owner; nothing is automatic.* A flagged provider is never dropped or down-weighted. Task 28
      Step 1 stops for the owner's ruling when any flag is present. For the current corpus (SkillCorner 909, GS 64,
      IDSSE 7 matches, each with hundreds of units per match) no flag is expected, but that is measured, not assumed.
  - These pooled values become the released base defaults, which also serve any provider absent from the map.

- [ ] **Step 1: Write the failing tests.**
  - Every reducer, both sides of every rule. Examples:
    - `welch_segment_rule(0.22, 0.83, 2700.0) == (400.0, True)` — R4's pin, where resolution needs ≥ 393.4 s and
      12 segments ≥ 8 holds.
    - A band so narrow that 8 segments cannot hold returns the resolution length with the flag `False`.
    - `first_acf_zero_crossing_s` on a sinusoid equals a quarter period.
    - `min_observed_fraction_rule` picks the smallest qualifying bin.
  - `test_driver_resumes_before_load_and_excludes`, on `tests/scripts/_fake_corpus.py` (the existing fake corpus):
    - a second run loads nothing;
    - a snapshot-provider match raises `MatchExcluded` and is recorded in `.excluded.json`.
  - `test_preflight_refuses_discarded_visibility` (the ADR-069 Layer 2 pre-flight).
  - `test_reduce_writes_derivation_and_codegen_reproduces`: under `tmp_path` and a monkeypatched `GENERATED_PATH`,
    the committed generator reproduces the written file byte-for-byte from `derivation.json`, with
    `BASE_SOURCE = "derivation"`.
  - `test_unit_weights` (both modes): the weights sum to 1; each match's units sum to its match share; with
    `provider_neutral=True` each provider sums to `1/P`.
  - `test_weighted_quantile_matches_numpy_inverted_cdf`, including equal-weight parity with unweighted
    `np.quantile(method="inverted_cdf")`.
  - `test_pooled_values_are_provider_neutral` (A3, TF58-PLAN-02):
    - On a synthetic two-provider corpus, provider A has 9 matches and cutoffs near 0.3 Hz; provider B has 1 match
      with 50 runs and cutoffs near 0.7 Hz. `pooled` equals the reducers applied under
      `unit_weights(provider_neutral=True)` exactly.
    - Non-vacuity, both directions: the match-weighted (corpus-representative) value and the run-weighted value both
      differ from the provider-neutral value on this corpus. The test asserts each difference, so it would fail if the
      implementation silently used either rejected weighting.
  - `test_per_provider_values_are_match_weighted`: one provider with one heavy match (50 runs) and nine light matches
    (5 runs each). The per-provider cutoff is the match-weighted median, not the run-weighted one (asserted
    different).
  - `test_provider_bootstrap_se_is_match_level_and_seeded`:
    - resampling is by match: a provider whose runs vary only **within** matches (identical match medians) has SE 0;
    - the same `seed_key` gives an identical SE, and a different key a different resample set.
  - `test_thin_provider_flagged_and_well_sampled_not` (TF58-PLAN-03), both sides on one synthetic corpus. There are
    three providers:
    - two well-sampled (20 matches each, low per-match dispersion);
    - one thin and planted noisy (2 matches with widely different per-match cutoffs).

    The thin one is `flagged`; the two well-sampled ones are `ok`. Non-vacuity: the pooled value is still the
    provider-neutral one (the flag never alters weights), asserted equal to the unflagged computation.
  - `test_thin_provider_not_assessable_with_one_provider`.
  - `test_refuses_dirty_tree_without_flag`.
  - `test_input_contract_declared_and_written`.
- [ ] **Step 2: Detector widening (C3), detection first.**
  - In `tests/scripts/test_input_contracts.py`, add `test_detector_scans_any_json_with_input_contract`: a planted
    `derivation.json` carrying `input_contract.driver = "d"` is found by `_artifacts_for("d", tmp_path)`. Run it:
    **FAIL**, because only `metrics.json` is scanned.
  - Then widen `_artifacts_for` to `root.rglob("*.json")` (same content filter). Run it: **PASS**. The existing
    plants (`FIRES` / `SILENT` / another driver) stay green.
- [ ] **Step 3: Run the driver tests** → **FAIL**.
- [ ] **Step 4: Implement the driver.** Enrol it in `ARTIFACT_DRIVERS` and `_DECLARING`.
- [ ] **Step 5: Run** the driver tests, `tests/scripts/test_provenance_wiring.py`, `test_input_contracts.py`,
  `test_corpus_driver_resilience.py`, `test_driver_load_hook.py` and `test_artifact_provenance_output.py` → **PASS**.
  Then ruff + format + pyright.

---

### Task 21: ruthless-efficiency 0.7.0 migration (D21) — needs 0.7.0 installed

**Precondition.** ruthless-efficiency 0.7.0 is on PyPI (confirmed 2026-09-26). Install it into **ragnarok's own venv
only**:
```
uv pip install --python .venv "ruthless-efficiency[optuna]==0.7.0"
```
Confirm that `python -c "import ruthless; print(ruthless.__version__, ruthless.__file__)"` prints `0.7.0` from
site-packages. Task 25 Step 1 re-confirms this and runs `uv lock`.

**Files:**
- Modify:
  - `pyproject.toml:74-86,127,138` (three floors + the comment; C17)
  - `scripts/_provenance.py` (the `objective_id` helper)
  - `silly_kicks/calibration/_spaces.py:25,48,52,73,95,116` (three builders + doctests at :10, :34, :58, :101)
  - `scripts/calibrate_tracking_defaults.py:58-72,406`
  - `scripts/calibrate_xt_bandwidth.py:38-43,377`
  - `scripts/check_stage1_argmax.py:91`
  - `scripts/train_xshot_occurrence.py:175-205` and its caller(s)
  - `scripts/train_xcross_attempt.py:250-273` and its caller(s)
  - `tests/tracking/test_xshot_occurrence_integration.py:69`
  - `tests/calibration/test_spaces.py` (all 7 calls, C15)
  - `tests/calibration/test_calibration_e2e.py:40,71`
  - `tests/calibration/test_cli_smoke.py:30`
  - `tests/calibration/test_calibrate_xt_bandwidth_cli.py:22,33`
  - `uv.lock` (after the PyPI release only)
- Test: `tests/scripts/test_provenance_objective_id.py`

**Interfaces — produces:**

```python
# scripts/_provenance.py
def objective_id(objective: object, inputs: Mapping[str, object], *, prov: Mapping[str, object] | None = None) -> str
    # "<module>.<qualname of the objective class>@<prov commit>:<inputs['digest']>"; the digest must come from
    # declare_inputs (else ValueError); a tree_state other than "clean" appends "+dirty-<uuid4 hex>" (C4)
# silly_kicks/calibration/_spaces.py -- builders gain a REQUIRED keyword-only `objective_id: str`
def stage1_config(*, n_trials: int, store_path: str, objective_id: str, sampler=...) -> OptunaConfig
def stage2_config(*, n_trials: int, store_path: str, objective_id: str, sampler=...) -> OptunaConfig
def xt_bandwidth_config(*, n_trials: int, store_path: str, objective_id: str, sampler=...) -> OptunaConfig
```

**Per-site identity rule (one rule, D21).**
- `inputs` = `declare_inputs(driver="<script stem>", args=<the parsed CLI arguments minus the resume knobs {store,
  n_trials, allow_dirty, out, list_matches}>, match_ids=<sorted ids the objective reads>, **site extras)`.
- The site extras:

  | Site | Extras |
  |---|---|
  | `calibrate_tracking_defaults` | `stage` |
  | trainers | the per-fold `tag` |
  | `check_stage1_argmax` | none: it builds a config only to read `metric`/`direction`, but still passes the helper's id, so no caller is special |

- The testable seams gain a required keyword, `objective_id: str`, forwarded to the builder or to `StoreConfig`:
  - `run_stage`;
  - `run_xt_bandwidth`;
  - the trainers' `_hpo_once`, whose inputs are passed in and completed with `tag`.
- Tests and doctests pass a literal (`objective_id="test-objective"`).

- [ ] **Step 1: Write the failing tests.**
  - `test_provenance_objective_id.py`:
    - `test_format_on_clean_tree` (with an injected `prov`).
    - `test_dirty_and_unknown_trees_get_a_unique_nonce`: two calls differ, and both contain `+dirty-`.
    - `test_requires_declared_digest`.
    - `test_class_and_instance_give_same_prefix`.
  - `tests/calibration/test_spaces.py`: `test_builders_require_objective_id` (omitting it → `TypeError`) and
    `test_builders_forward_objective_id_to_store`.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement.** Then run `git grep -n "StoreConfig(\|stage1_config(\|stage2_config(\|xt_bandwidth_config("`
  and confirm **every** hit passes an id (the sweep is the floor).
- [ ] **Step 4: Set the pins.**
  - Set every `ruthless-efficiency[optuna]` requirement, in `[calibration]`, `[test]` and `[train]`, to exactly
    `>=0.7.0`.
  - **Remove any `<0.7.0` ceiling.** The F1b cycle's CI hotfix caps ruthless at `>=0.6.0,<0.7.0` because 0.7.0's
    required `objective_id` broke `main`'s CI (plan-review r2, *outside this round*). The cap was not on `origin/main`
    when r2 was written (`git show origin/main:pyproject.toml` still shows `>=0.6.0`), but it may be by the time this
    task runs.
  - TF-58 is the deliberate 0.7.0-adoption cycle the cap defers to. The ceiling lift and this task's whole `objective_id`
    migration (every `StoreConfig(...)` site and every builder caller) therefore land in the **same** commit (commit 1).
    A lift without the migration re-breaks CI; a migration without the lift cannot install.
  - Verify: `git grep -n "ruthless-efficiency" -- pyproject.toml` shows only `>=0.7.0`, with no `<` anywhere.
  - Rewrite the floor comment at `pyproject.toml:74-84`. State that 0.7.0 is the first release with
    `GridSearchStrategy` and a required `StoreConfig.objective_id`, and that the migration landed together with the
    ceiling removal.
  - Run `uv lock` (0.7.0 is on PyPI).
- [ ] **Step 5: Run** `tests/calibration tests/scripts/test_provenance_objective_id.py
  tests/tracking/test_xshot_occurrence_integration.py tests/scripts -m "not e2e"` → **PASS**. Then ruff + format +
  pyright.
- [ ] **Step 6: Legacy-store notice.** Draft the CHANGELOG lines (they land in commit 2): existing owner-held Optuna
  studies (DGX) now raise on resume. The remedy is `ruthless.strategies.optuna_.adopt_legacy_store(config)` when the
  objective is genuinely unchanged, else a new store path (§12). Keep the draft in the ADR draft until Task 28.

---

### Task 22: D2 `scripts/calibrate_coordination.py` (Tier C, §8.4) — needs 0.7.0 installed

**Files:**
- Create: `scripts/calibrate_coordination.py`
- Modify: `tests/scripts/test_provenance_wiring.py` (`ARTIFACT_DRIVERS`), `tests/scripts/test_input_contracts.py`
  (`_DECLARING`)
- Test: `tests/scripts/test_calibrate_coordination.py`

**Constants (C26 — level definitions).** §8.4 fixes the level counts and says only `vc_epsilon` is "multiples of the
derived value". The plan makes every level relative to each provider's own Tier-B value. That lets one grid serve
every provider and makes the OAT baseline identical across providers:

```python
SWEEP: dict[str, tuple[float, ...]] = {
    "butterworth_cutoff_hz": (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0),   # x Tier-B (7)
    "max_detection_gap_s":   (0.5, 0.75, 1.0, 1.5, 2.0),              # x Tier-B (5)
    "min_observed_fraction": (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3),        # + Tier-B, clipped to [0, 1] (6)
    "welch_segment_s":       (0.5, 0.75, 1.0, 1.5, 2.0),              # x Tier-B (5)
    "vc_epsilon":            (0.5, 0.75, 1.0, 1.5, 2.0),              # x Tier-B (5)
}
BASELINE = {"butterworth_cutoff_hz": 1.0, "max_detection_gap_s": 1.0, "min_observed_fraction": 0.0,
            "welch_segment_s": 1.0, "vc_epsilon": 1.0}
PREPARATION_PARAMS = ("butterworth_cutoff_hz", "max_detection_gap_s")
AFFECTED_FAMILIES = {"butterworth_cutoff_hz": COORD_METHOD_FAMILIES, "max_detection_gap_s": COORD_METHOD_FAMILIES,
                     "min_observed_fraction": COORD_METHOD_FAMILIES, "welch_segment_s": ("coherence",),
                     "vc_epsilon": ("vector_coding",)}
```

Every level, baseline and point is a native Python `float` (§8.4 level types). A test pins `type(v) is float`.

**Layers:**
- **Layer (a), `--layer a --level <param>=<value>`** (one invocation per preparation level: 7 + 5 − 1 shared
  baseline = 11 passes).
  - A `for_each` over the corpus with `n_surrogates=0` and `match_tables(..., variants=...)`.
  - On the **baseline** preparation level, the variants include every post-preparation level (6 + 5 + 5, deduped),
    so post-preparation levels reuse the prepared signals (§8.4).
  - The `token_inputs` include the level, the variant list, the D1 derivation digest and `input_contract()`.
- **Layer (b), `--layer b`: OAT selection on ruthless.**
  - `CoordinationReliabilityObjective` implements `ruthless.objective.Objective`: `evaluate(candidate) -> Metrics`.
  - It identifies the single deviating parameter (OAT) and reads that level's shards (a missing shard raises
    `ruthless.errors.FatalEvaluationError`, §8.4).
  - For each `match_cv_splits` fold (on the corpus match ids) it computes the mean ICC(1) (`scripts._reliability.icc1`,
    groups = team) over the `AFFECTED_FAMILIES` metric columns on the fold's **test** matches. The ICC is per
    provider stratum, weighted by the stratum's match count.
  - It returns `{"reliability": mean over folds, "reliability_se": cv_standard_error(folds), "fold_00": ..., ...}`.
  - The run is `GridSearchStrategy(GridConfig(kind="grid", metric="reliability", direction="maximize",
    design="one_at_a_time", baseline=BASELINE, param_space={k: {"kind": "choice", "choices": v} for k, v in
    SWEEP.items()}, store=StoreConfig(kind="sqlite", path=str(out / "grid_oat.db"), objective_id=<C27 id>)))`, run
    with `.run(objective, backend=InProcessBackend())`.
  - Per parameter: `select_recommended_point(incumbent=<baseline PointScore>, candidates=<that parameter's
    PointScores>)`, which applies `MIN_EFFECT_SIZE` and `exceeds_noise_floor` (the paired SE) (C14).
- **Confirmation, `--layer confirm`.**
  - The joint point is the baseline with every moved parameter at its selected level.
  - If a preparation parameter moved, a layer-(a) pass at the joint preparation level runs first.
  - Then `GridConfig(design="points", points=[joint], store=StoreConfig(path=str(out / "grid_confirm.db"),
    objective_id=<C27 id of the confirm generations>))` — a different path (§8.4).
  - The gate: `select_recommended_point(incumbent=baseline, candidates=[joint]).moved` **and** `gated_pass(
    evaluate_hypotheses(<joint tables>))`. If it clears, the codegen renders `render_generated_params(derivation,
    calibration)`. That applies the moved selections to the per-provider values **and** to the pooled base (A3).
    Otherwise the Tier-B values stand, and the fallback and its reason are recorded.
- **C27 (the objective identity across passes).** §8.4 sets `objective_id` to "the shard-generation token". D2's
  objective reads several generations (one per preparation level), so the id is
  `"calibrate_coordination:" + sha256("|".join(sorted(generation tokens read))).hexdigest()[:16]`. Any regenerated
  shard set changes the id, and ruthless then refuses to resume from stale scores.
- **Output:** `docs/research/tf58_team_coordination/calibration.json` holds:
  - per parameter: the levels, the `PointScore`s, the `Selection` (moved, reason, effect size, paired SE);
  - the joint point, the confirmation result and the gate outcome;
  - the C27 ids;
  - `input_contract`, `run_commit`, `run_tree_dirty` and the stage timings (C14).

- [ ] **Step 1: Write the failing tests.** All use synthetic shards written under `tmp_path`; nothing touches the
  corpus.
  - `test_levels_are_native_floats_and_include_baseline`.
  - `test_level_counts_match_spec` (7, 5, 6, 5, 5).
  - `test_oat_config_constructs_and_enumerates_1_plus_sum_levels_minus_1`.
  - `test_objective_reads_the_deviating_level` (a planted shard per level; the objective returns that level's ICC).
  - `test_missing_shard_raises_fatal`.
  - `test_selection_moves_only_beyond_noise_floor` (both sides).
  - `test_confirmation_gate_requires_both_reliability_and_hypotheses` (four combinations).
  - `test_fallback_keeps_tier_b_and_records_reason`.
  - `test_objective_id_changes_when_any_generation_changes` (C27).
  - `test_oat_and_confirm_use_different_store_paths`.
  - `test_resume_from_store_skips_evaluated_points`: a second `GridSearchStrategy` run over the same store and id
    evaluates nothing (`result.diagnostics["n_from_store"]`).
  - `test_calibration_artifact_fields`.
  - `test_refuses_dirty_tree_without_flag`.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement.** Enrol the driver in `ARTIFACT_DRIVERS` and `_DECLARING`.
- [ ] **Step 4: Run** the driver tests plus the provenance, input-contract and resilience gates → **PASS**. Then ruff +
  format + pyright.

---

### Task 23: D3 `scripts/validate_team_coordination.py` (the in-cycle artifact, §8.5)

**Files:**
- Create: `scripts/validate_team_coordination.py`
- Modify: `tests/scripts/test_provenance_wiring.py`, `tests/scripts/test_input_contracts.py`
- Test: `tests/scripts/test_validate_team_coordination.py`

**Passes.** Each is one `for_each` with its own generation:
1. `--pass metrics`:
   - final params (`CoordinationParams.for_provider`) with `n_surrogates=199`;
   - per-match tables for all 7 families plus the windows;
   - per-stage timings (R8).
2. `--pass stoppage` (GS and IDSSE only, D20): per match, the metric tables under each of `stoppage_evidence ∈
   {"ball_state", "events", "none"}` (C12). It also records the event-derived vs `ball_state` intervals longer than
   `STOPPAGE_LEG_MIN_S`: precision and recall (an event interval is a hit when it overlaps any ball-state interval),
   and the mean IoU of the matched pairs.
3. `--pass occlusion` (GS and IDSSE): the final occlusion error curves at the calibrated W (from `derivation.json`).
4. `--pass reduce`. It writes `docs/research/tf58_team_coordination/metrics.json` (with `input_contract`) and
   `report.md`, containing:
   - `evaluate_hypotheses` (H1–H7) with every statistic and pass flag;
   - per metric and provider: `icc1`, `split_half_reliability` and `type_ii_slope` (`scripts/_reliability.py`);
   - `compare_providers` poolability;
   - SkillCorner coverage stratification: each metric's quantiles by `coord_observed_fraction` decile;
   - the stoppage leg: precision, recall, IoU, and each metric's change between the three splitting modes;
   - the occlusion curves;
   - real-data liveness: every metric column non-NaN and non-constant somewhere in the corpus. This assertion is
     **reported**; if R2 is declined it also backs spectral and coherence liveness (Task 18 Step 7);
   - the stage timing summary against the §7.15 cost model.

   A failed hypothesis is recorded as a finding for the ADR (§8.5), never dropped.

- [ ] **Step 1: Write the failing tests** (synthetic per-match shards under `tmp_path`):
  - `test_reduce_reports_every_hypothesis_with_threshold_refs`: values come from `_coordination_thresholds`, never
    literals. This is an AST check that the driver has no numeric literal equal to a threshold constant.
  - `test_stoppage_leg_precision_recall_iou` (planted intervals, both hit and miss).
  - `test_coverage_stratification_deciles`.
  - `test_reliability_block_uses_shared_module` (`_reliability` functions are called; identity via monkeypatch).
  - `test_liveness_block_flags_a_dead_column`.
  - `test_failed_hypothesis_recorded_not_dropped`.
  - `test_resume_before_load_and_exclusion` (the fake corpus).
  - `test_refuses_dirty_tree_without_flag`.
  - `test_input_contract_written`.
- [ ] **Step 2: Run** → **FAIL**.
- [ ] **Step 3: Implement.** Enrol the driver in `ARTIFACT_DRIVERS` and `_DECLARING`.
- [ ] **Step 4: Run** the driver tests plus the provenance, input-contract and resilience gates → **PASS**. Then ruff +
  format + pyright.

---

## Documentation, commit 1, rebase, DGX, commit 2 (Tasks 24–28)

### Task 24: Real-match cost measurement and the commit-1 documentation

**Files:**
- Create: `docs/superpowers/adrs/ADR-111-team-coordination.md`
- Modify:
  - `AGENTS.md` (one Architecture bullet)
  - `docs/context/tracking-metrics.md` (a TF-58 section)
  - `docs/superpowers/specs/2026-09-26-tf58-team-coordination-design.md` (the status line only)

- [ ] **Step 1: Measure one real match per provider** (§7.15: "before any corpus launch").
  - Run `.venv/Scripts/python scripts/validate_team_coordination.py --pass metrics --providers <p> --max-matches 1
    --allow-dirty --out <scratchpad>/cost_<p>` for `skillcorner` (10 Hz), `idsse` (25 Hz) and `gradientsports`.
  - Needs the pining token (`--token` or its env var). If the token is not available in the session, ask the owner to
    run the three commands with `! <command>`.
  - Read the manifest's per-stage timings, once with numba and once with `SILLY_KICKS_COORDINATION_FORCE_NUMPY=1`.
  - Compare with §5 goal 6 (≤ 45 s per match numpy-only, ≤ 15 s with numba). A miss is reported to the owner with the
    stage breakdown **before** any further step. Never trim a method to fit the budget without an owner ruling.
- [ ] **Step 2: Write the ADR draft**, following `docs/superpowers/adrs/ADR-TEMPLATE.md`, design section only:
  - context;
  - decisions D1–D21 (a pointer to spec §4) plus the owner rulings R1–R5 as ruled, with amendments A1–A3:
    - A1: the `computed_nonconverged` surrogate token;
    - A2: the exact orientation predicate (Shewchuk 1997, cited in `NOTICE` beside the TF-58 paragraph);
    - A3: pooled derived base defaults and the `BASE_SOURCE` gate;
  - the concretisations C1–C27 that change behaviour (C1, C3, C4, C12, C13, C17, C19, C20, C22, C24–C27);
  - consequences;
  - performance (Task 0 Step 4 vs Task 3 Step 8; Step 1 above);
  - the ADR-048 amendment (the `Unit` vocabulary, R5) and the ADR-098 amendment (completeness keyed per exported
    constant).

  The results section is written in commit 2.
- [ ] **Step 3: `AGENTS.md`.** Add one Architecture bullet of at most 600 characters (the `test_agents_md_budget.py`
  cap), for example:

  > **Coordination** (`coordination/`): `compute_team_coordination`/`build_coordination_signals` + 7 family computes
  > (relative phase, xcorr, vector coding, spectral, coherence, cluster phase+SampEn, RSI) with seeded surrogate
  > baselines; tracking-only, not action-coupled; windows `period_windows`/`possession_windows_from_*`; collective
  > kernel `compute_collective_variables` single-sources team shape + back line (ADR-NNN). →
  > `docs/context/tracking-metrics.md`.

  Run `tests/test_agents_md_budget.py`: AGENTS.md is 24,946 bytes against a ceiling of 27,750.
- [ ] **Step 4:** Add a narrative section to `docs/context/tracking-metrics.md`: the why of each method, the
  divergences table (§7.8.8), C24 orientation and the D20 stoppage precedence.
- [ ] **Step 5:** Set the spec status line to `**Status:** Approved (owner, 2026-09-26; independent reviews r1–r3).`
  Change nothing else in the spec.
- [ ] **Step 6:** Run `tests/test_agents_md_budget.py`, `tests/test_c4_*.py` and the doctest job → PASS.

---

### Task 25: Commit-1 gate and the approval stop

- [ ] **Step 1: ruthless 0.7.0 on PyPI** (§13 item 1). Confirm with `uv pip index versions ruthless-efficiency` (or the
  PyPI page) that `0.7.0` exists. Then:
  - `uv lock`;
  - `uv pip install --python .venv "ruthless-efficiency[optuna]==0.7.0"`, which replaces the editable install;
  - confirm that `python -c "import ruthless; print(ruthless.__version__, ruthless.__file__)"` shows `0.7.0` from
    site-packages;
  - re-run `tests/calibration tests/scripts -m "not e2e"`.

  If 0.7.0 is not on PyPI, **stop**. Commit 1 is not proposed before it is (§13 item 1).
- [ ] **Step 2: Rebase state.** If F1b (ADR-106) or native DAS (ADR-107/108) has merged into `main` since `05cfa56`,
  run Task 26 first.
- [ ] **Step 3: Full gate** (Global Constraints). Run both pandas legs, lint at CI scope, bare pyright and the doctest
  job. Compare the failures with Task 0's baseline list: a new failure blocks, and a pre-existing one is listed in the
  approval request.
- [ ] **Step 4: `/final-review`** (the owner's skill). Fix its findings. A finding that would change scope goes to the
  owner.
- [ ] **Step 5: Number the ADR.** Rename `ADR-111-team-coordination.md` to `ADR-NNN-tf58-team-coordination.md`,
  where NNN is the next free number on `main` at this moment (§16). Update every `ADR-NNN` reference: the ADR, the C4
  description, `AGENTS.md` and `docs/context`. Re-render the C4 model with `dot` (Task 18 Step 11) if its text
  changed.
- [ ] **Step 6: Tree audit.**
  - `git status --short` must list exactly the files of this plan's file map (commit-1 subset), the spec and this plan.
  - Anything else (a stray scratch file, a lockfile from another tool) is removed or explained **before** asking.
  - `uv.lock` is included, since it changed in Step 1.
- [ ] **Step 7: The approval stop.** Show the owner:
  - `git status --short`;
  - `git diff --stat`;
  - the untracked-file list;
  - the proposed commit message (below);
  - the proposed PR title and body (draft PR; it stays open for commit 2, §13 item 5).

  Ask for explicit approval of **this commit + push + PR**. Do nothing until the owner answers. Never create the
  approval sentinel.
- [ ] **Step 8 (only after approval):** `git add` of the explicit path list (never `git add -A`); `git commit`;
  `git push -u origin feat/tf58-team-coordination`; `gh pr create --draft`, with the approved title and body.

Proposed commit-1 message (adjust only the ADR number):

```
feat(coordination): TF-58 team-coordination dynamics + vectorised collective kernel (ADR-NNN)

New silly_kicks.coordination: Hilbert relative phase, lagged cross-correlation, vector coding,
spectral median frequency, Welch coherence, cluster phase + SampEn/Cross-SampEn and the relative
stretch index, at team-team / cross-variable / intra-team / dyad levels over period, sliding and
possession windows, each with a seeded time-shift (or IAAFT) surrogate baseline. Seven metric
contracts. Detection-aware SkillCorner handling; D20 dead-ball evidence precedence.

Seams: tracking.preprocess gains a zero-phase Butterworth (Winter dual-pass correction), uniform
resampling and residual analysis; tracking._collective single-sources compute_team_shape (hull
<= 1e-9 rel) and compute_defensive_line (byte-identical). Drivers D1/D2/D3 (derive, calibrate on
ruthless GridSearchStrategy, validate). ruthless-efficiency >= 0.7.0 (StoreConfig.objective_id
migration). Artifacts and generated params follow in the next commit.
```

The trailer is the one mandated for the implementing session.

---

### Task 26: Rebase onto F1b and native DAS (§13.1) — runs when either merges into `main`

- [ ] **Step 1: Tree state.** `git fetch origin`, then confirm the tree is clean (untracked spec and plan excepted if
  commit 1 has not happened yet).
- [ ] **Step 2: Rebase.** `git rebase origin/main`. Resolve conflicts in the shared registries:
  - `NOTICE`, the C4 files, `AGENTS.md`, `CHANGELOG`;
  - `ARTIFACT_DRIVERS`, `.test_durations`, `pyproject.toml`;
  - the numba cache-key line (keep **both** sides' patterns);
  - `tests/test_public_api_examples.py`, `tests/_scale_guarded.py`;
  - **the ruthless pins.** If `main` now carries the F1b CI hotfix's `>=0.6.0,<0.7.0` ceiling, resolve every pin to
    `>=0.7.0` (ceiling removed). Confirm with `git grep -n "StoreConfig(\|stage1_config(\|stage2_config(\|xt_bandwidth_config("`
    that every call site on the merged tree passes `objective_id`, including any site `main` added since `05cfa56`.
    Then re-run Task 21 Step 5. Lifting the ceiling without the complete migration re-creates the CI break the cap
    fixed.
- [ ] **Step 3: F1b.**
  - Every `team_id`-keyed `groupby` uses `observed=True`; `git grep -n "groupby(" silly_kicks/coordination
    silly_kicks/tracking/_collective.py` checks this.
  - Re-verify the upcast gate's scan scope on the merged `main` (the §3.5 claim: non-recursive `tracking/*.py` +
    `tracking/pitch_control/*.py`) by reading `tests/tracking/test_frame_coord_upcast_gate.py`. Then extend
    `_KERNEL_DIRS` with `silly_kicks/coordination` and `silly_kicks/tracking/preprocess`.
  - `smooth_frames(method="butterworth")` writes `x_smoothed`/`y_smoothed` at the storage dtype (F1b's cast).
  - Collective-kernel reads stay `.to_numpy(dtype="float64")`.
  - The parity tests (Task 1/3) run on F1b's float32 frames. Byte-identity is between the legacy oracle and the new
    kernel on the **same** frames, so it still holds.
- [ ] **Step 4: Native DAS.** If a reusable frame-contract validator (duplicate-row / ball-row checks) landed on `main`,
  `_signals.py` uses it instead of its own checks. Player ids stay category-safe (`id_compat`, `observed=True`).
- [ ] **Step 5: Sweep.** Re-run Task 0 Step 5's sweep on the merged tree. Every new caller of a changed seam gets
  evidence of both sides, per the §7.16 rule.
- [ ] **Step 6: Full gate**, both pandas legs.
- [ ] **Step 7: Approval** (Global Constraints exception (a)). If commit 1 was already pushed, the rebase needs a
  force-push. Show the owner the rebased
  `git log --oneline main..HEAD` and `git diff --stat origin/feat/tf58-team-coordination`, and ask explicit approval
  for `git push --force-with-lease`.

---

### Task 27: DGX runbook (owner-run, §13 item 3)

These commands are for the owner on the DGX, against the **clean commit-1 tree**. The drivers refuse a dirty tree,
and every artifact stamps `run_commit` and `run_tree_dirty: false`.

1. **Environment.**
   - `git clone` / `git fetch` and `git checkout <commit-1 SHA>`; `git status --short` must be empty.
   - `uv sync --extra kloppy --extra calibration --extra numba` (or the DGX's existing environment recipe for corpus
     drivers).
   - `export SILLY_KICKS_CORPUS_CACHE_DIR=<cache>` plus the pining token variable.
2. **Worker slices + the corpus list.** Write per-worker `match_ids_w<k>.json` from
   `scripts/_partition.list_match_ids(["skillcorner", "gradientsports", "idsse"])` split into 16 disjoint slices, AND
   the full unsplit listing as `corpus.json`.
   - Every per-worker pass takes `--match-ids-json match_ids_w<k>.json --corpus-json corpus.json`.
   - Every step that reads a corpus-wide result (D1 pass b / occlusion / reduce, D2 layer b / confirm, D3 reduce, the
     numerics reduce) takes `--corpus-json corpus.json`. It combines every worker's share and refuses unless the
     shares cover exactly that corpus, on one commit and one shard generation, with no failed match (B-1, §8.3).
   - Use one fresh `--out` per driver run, shared by all of that driver's workers. Nothing is written into the repo:
     each driver writes its artifacts into its own `--out` (owner ruling M-5, 2026-10-02).
3. **D1** (`--out $D1`).
   - `python scripts/derive_coordination_params.py --pass a` (16 workers);
   - then `--pass b` (16 workers; the per-provider cutoffs come from every worker's pass a);
   - then `--pass occlusion-cal` (16 workers; FOV-width calibration histograms);
   - then `--pass occlusion` (16 workers; ONE width from every worker's histogram);
   - then `--pass reduce` (single). It writes `$D1/derivation.json` and `$D1/_provider_params_generated.py`.
4. **D2** (`--out $D2 --derivation $D1/derivation.json`).
   - `python scripts/calibrate_coordination.py --layer a --level <level>` for each of the 11 levels in
     `calibrate_coordination.preparation_levels()` (`baseline` + the off-baseline preparation levels; 16 workers
     each);
   - then `--layer b` (single): combines all 11 levels, runs the OAT grid, writes `$D2/oat.json`;
   - then `--layer joint` (16 workers): the joint point's preparation pass when it moves >= 2 parameters, a no-op
     otherwise;
   - then `--layer confirm` (single). It writes `$D2/calibration.json` and `$D2/_provider_params_generated.py`: the
     calibrated module when the gate clears, else byte-identical to D1's.
5. **D3** (`--out $D3 --derivation $D1/derivation.json --calibration $D2/calibration.json`).
   `python scripts/validate_team_coordination.py --pass metrics`, then `--pass stoppage`, then `--pass occlusion`
   (16 workers each), then `--pass reduce` (single). The reduce writes `$D3/metrics.json` and `$D3/report.md`.
6. **Numerics no-flip gate** (`--out $NF`). `python scripts/validate_coordination_numerics.py --pass map` (16 workers),
   then `--pass reduce` (single). The reduce writes `$NF/numerics_noflip.json`.
7. **Return.** Copy `$D1/derivation.json`, `$D2/calibration.json`, `$D3/metrics.json`, `$D3/report.md` and
   `$NF/numerics_noflip.json` into `docs/research/tf58_team_coordination/`, and `$D2/_provider_params_generated.py`
   into `silly_kicks/coordination/` (commit 2). Check that:
   - each artifact's `run_commit` equals the commit-1 SHA and `run_tree_dirty` is `false`;
   - `calibration.json`'s `derivation_sha256` is the sha256 of the copied `derivation.json`, and D3's population
     block records the same derivation and calibration digests;
   - every `population` block says `population_checked_against: corpus_json` and `n_failed: 0`.

Expected wall time (§7.15): D3 ≈ 15–45 min on 16 workers. D1 and D2 are dominated by preparation, about 14 passes of
similar cost. The per-stage timings are in each manifest.

---

### Task 28: Commit 2 — artifacts, results, release bookkeeping

**Files:**
- Add:
  - `docs/research/tf58_team_coordination/{derivation.json, calibration.json, metrics.json, report.md,
    numerics_noflip.json}` (copied from the drivers' `--out` folders, Task 27 step 7)
  - `silly_kicks/coordination/_provider_params_generated.py` (copied from `$D2`, Task 27 step 7)
- Modify:
  - `tests/coordination/test_config.py` (`test_generated_map_empty_in_commit_1` becomes
    `test_generated_map_reproduces_from_artifacts`)
  - `tests/coordination/test_provider_params_generated.py`. Create it with two tests:
    - `render_generated_params(json(derivation), json(calibration))` equals the committed file byte-for-byte (§8.2);
    - `test_base_defaults_are_derived` (A3): `BASE_SOURCE == "derivation"`, and `CoordinationParams()` Tier-B values
      equal `derivation["pooled"]` with the calibration selections applied. It fails if an interim placeholder would
      ship.
  - `tests/coordination/test_config.py`: `test_interim_base_matches_R4_in_commit_1` is deleted. It is superseded by
    `test_base_defaults_are_derived`, and `INTERIM_BASE` stays pinned by Task 19's codegen test.
  - the ADR (results section: H1–H7 outcomes, reliability, poolability, the occlusion and stoppage legs, the cost
    summary, every failed hypothesis as a finding for the owner)
  - `CHANGELOG.md`
  - `TODO.md` (TF-58 row removed)
  - `silly_kicks/_version.py`
  - `.test_durations`

- [ ] **Step 1:** Place the DGX outputs.
  - **First, read `derivation.json`'s `thin_providers` block (TF58-PLAN-03).** If any provider is `flagged` for any
    quantity, stop. Show the owner each flagged quantity (the provider value, its SE, the between-provider SD) and wait
    for a ruling before going on. The ruling is recorded in the ADR results section.
  - Then run `tests/coordination tests/scripts -m "not e2e"` → `test_provider_params_generated.py` goes red until
    Step 2.
- [ ] **Step 2:** Write the two reproduction tests. The `test_input_contracts.py` detector must be **silent** on the
  new artifacts: their digests match live code.
- [ ] **Step 3: CHANGELOG.** Add the entry keyed by the next `PR-S` number, containing:
  - Added: the package, the seams, the drivers.
  - Changed:
    - `convex_hull_area` / `team_shape_convex_hull_area_*` move by ≤ 1e-9 relative (≤ 1e-9 m² absolute on
      precision-flat frames, R3). They are exactly 0.0 if and only if the frame is exactly collinear, decided by an
      adaptive exact orientation predicate (A2);
    - `compute_defensive_line` is byte-identical;
    - `PreprocessConfig` gains two fields;
    - the calibration builders require `objective_id`.
  - Breaking:
    - the three builders' new required keyword;
    - ruthless ≥ 0.7.0;
    - **owner-held Optuna studies become legacy** — the remedy is `adopt_legacy_store(config)` or a new store path
      (§12, Task 21 Step 6).
  - A downstream notice for the lakehouse: the seven tables, their grains, and the SkillCorner coverage caveat.
- [ ] **Step 4: Version.** Bump `silly_kicks/_version.py` to the next minor after `main`'s current version at this
  moment (§16; F1b and DAS claim theirs first).
- [ ] **Step 5: `.test_durations`** (ADR-074: CI-measured, never local). This is Global Constraints exception (b). It
  follows the procedure in `docs/context/ci.md` ("temporarily re-add a `durations-capture` job …").
  - **Ask** the owner for one explicit approval covering all four outward actions:
    1. create the throwaway branch `ci/tf58-durations-capture` from the feature branch's working state, with one
       commit adding the temporary `durations-capture` job (full `pytest -m "not e2e" --store-durations` under the
       warm numba cache; upload with `include-hidden-files: true`, since `.test_durations` is a dotfile);
    2. push it;
    3. open a **draft PR to `main`**. `ci.yml:3-7` triggers CI only on `pull_request` to `main` or a push to `main`,
       so a bare branch push would never run the job;
    4. after the artifact is downloaded, close the PR unmerged and delete the branch, locally and on the remote.
  - After approval: run the four actions, then download the `test-durations-ci` artifact
    (`gh run download <run-id> -n test-durations-ci`).
  - Copy the downloaded `.test_durations` over the feature branch's working copy. It lands in commit 2 (Step 7);
    nothing is committed on the feature branch here.
  - Check out the feature branch again and confirm `git status --short` shows only commit-2 files plus
    `.test_durations`.
- [ ] **Step 6: Full gate**, both pandas legs, then `/final-review`.
- [ ] **Step 7: The approval stop.** Show `git status --short`, `git diff --stat` and the commit message, and ask
  explicit approval for **commit + push**. After approval: explicit `git add` paths, `git commit`, `git push`. Mark the
  PR ready for review only with the owner's go. Never merge before CI is green (§13 item 5).

---

## Spec → plan test mapping

| Spec (§9 clause) | Plan test(s) |
|---|---|
| 9.1 phase: known offset, anti-phase, noise lowers R, padding non-vacuity | T8 `test_known_offset_sinusoids`, `test_anti_phase_is_180`, `test_noise_lowers_R_monotonically`, `test_reflect_padding_reduces_edge_error` |
| 9.1 circular stats parity; histogram edges incl. ±180° wrap | T8 `test_circular_summary_parity_with_scipy`, `test_histogram_edges_both_sides`, `test_near_in_phase_threshold_both_sides` |
| 9.1 xcorr: +k shift → +k/fs, r = 1; inverted → negative; Fisher single slice | T9 `test_shifted_copy_positive_lag_when_a_leads`, `test_inverted_b_negative_r`, `test_fisher_single_slice_identity`, `test_matches_numpy_corrcoef_per_lag` |
| 9.1 vector coding: every Table 1 edge; ε stationary; printed Eq. 2 pin; no span across a split | T10 `test_table1_edges_both_sides`, `test_stationary_with_epsilon_both_sides`, `test_printed_eq2_is_wrong` (C19); T17 `test_vector_coding_differences_never_span_a_split` |
| 9.1 spectral: pure tone; offset invariance; coherence ≈ 1 / ≈ 1/K | T11 `test_pure_tone_median_frequency`, `test_positive_offset_does_not_move_median`, `test_coherence_near_one_for_linearly_filtered_pair`, `test_coherence_near_inverse_k_for_independent_noise`, `test_pools_spectra_not_coherences` |
| 9.1 cluster: identical → 1; constant lags → ρ_group 1; random → small; `min_players` both sides | T12 `test_identical_phases_rho_one`, `test_constant_per_player_lags_rho_group_one`, `test_uniform_random_phases_rho_small`, `test_min_players_both_sides` |
| 9.1 SampEn: published reference; exact counter parity; `entropy_undefined` | T12 `test_sampen_gaussian_white_noise_matches_published_analytic_value`, `test_counters_exact_against_naive`, `test_numba_parity.py`, `test_entropy_undefined_reachable` |
| 9.1 surrogates: autocorrelation; percentile separation; order-invariant seeds; accelerated == direct | T13 `test_time_shift_preserves_autocorrelation_exactly`, `test_coupled_pair_percentile_separates_from_uncoupled`, `test_seed_independent_of_processing_order`, `test_accelerated_*_equals_direct` |
| 9.1 Butterworth: zero phase; −3 dB dual pass; residual recovers cutoff; resample exact; never across a split | T4 `test_zero_phase_lag_on_in_band_sinusoid`, `test_dual_pass_minus_3db_at_cutoff`, `test_residual_analysis_recovers_planted_cutoff`, `test_resample_uniform_exact_on_linear`, `test_resample_never_crosses_split` |
| 9.2 `compute_defensive_line` byte-identical (fixtures, both directions, n variants, < 3 players) | T1/T3 `test_defensive_line_byte_identical_to_legacy`, `test_parity_fixtures_contain_rtl_small_and_adaptive_groups`, `test_parity_fixtures_contain_cut_ties` |
| 9.2 `compute_team_shape` (all but hull exact; hull 1e-9; collinear 0; n < 3 NaN; rtl) | T1/T3 `test_team_shape_matches_legacy`; T2 `test_hull_area_exact_collinear_and_coincident_is_zero`, `test_hull_area_fewer_than_three_is_nan` |
| 9.2 hull vs ConvexHull adversarial; spread identity | T2 `test_hull_area_matches_qhull_random`, `test_hull_area_adversarial`, `test_hull_area_precision_flat_within_abs_bound` (R3), `test_spread_identity_matches_naive_double_sum`; A2 `test_exact_predicate_rejects_float_false_positive`, `test_exact_predicate_accepts_exactly_collinear_dyadic_sets`, `test_exact_predicate_agrees_with_rational_oracle`, `test_exact_fallback_runs_only_on_unsettled_rows` |
| 9.2 `add_team_shape` golden | T3 `test_add_team_shape_only_hull_area_moves` (+ `test_restdefense_output_unchanged`) |
| 9.3 stoppage and detection-gap splits both sides | T16 `test_stoppage_splits_segments_both_sides`, `test_detection_gap_split_both_sides`; T15 `test_only_longer_than_max_stoppage_splits` |
| 9.3 stoppage precedence (each reachable; unclassified raises; constant-alive SkillCorner never `ball_state`; every restart type + goals) | T15 `test_stoppage_precedence_each_source_reachable`, `test_event_intervals_for_every_restart_type_and_goals`, `test_dead_ball_observed_unclassified_provider_raises`; T5 taxonomy tests |
| 9.3 substitution without splicing; red-card counts | T16 `test_substitution_no_splice`, `test_red_card_splits_team_segment_and_steps_count` |
| 9.3 each builder incl. terminal events; mixed sources refused | T15 builder tests, `test_possession_windows_end_at_terminal_event`, `test_mixed_possession_sources_refused`, `test_caller_mixed_with_builder_refused` |
| 9.3 every §7.3 refusal | T16 `test_every_refusal` |
| 9.3 every `COORD_SOURCE_VALUES` token reachable; report conservation | T17 `test_every_source_token_reachable`, `test_every_surrogate_token_reachable` (R1), `test_report_conserves_windows_and_rows` |
| 9.3 warning category distinct | T14 `test_warning_category_is_distinct`; T17 `test_single_warning_from_orchestrator_both_sides_of_threshold` |
| 9.3 purity; id-dtype invariance | T18 `test_purity.py`, `test_id_dtype_invariance.py`; T3 `test_does_not_mutate_input`, `test_id_dtype_invariance` |
| 9.3 mirror and identity invariance | T18 `test_orientation_invariance.py` (with the C24 non-vacuity) |
| 9.4 `tests/coordination/__init__.py` | T8 |
| 9.4 glossary + NOTICE tokens + `Unit` `cycles/min` (+ R5 `switches/min`) | T18 Step 9; `tests/test_feature_glossary_coverage.py`, `test_feature_glossary_notice_linkage.py` |
| 9.4 run-and-diff legs + non-vacuity anchors | T18 Step 9 (`_coordination_columns`, `test_each_leg_is_non_vacuous`) |
| 9.4 `_PUBLIC_MODULE_FILES` + Examples everywhere | T18 Step 10; `tests/test_public_api_examples.py` |
| 9.4 C4 container + glossary count, rendered with `dot` | T18 Step 11; `test_c4_dsl_description_cap.py`, `test_c4_feature_column_count.py` |
| 9.4 `ARTIFACT_DRIVERS` enrolment of D1–D3 | T20/T22/T23; `tests/scripts/test_provenance_wiring.py` |
| 9.4 import allowlist + planted violations | T18 Step 2 |
| 9.4 `metric_contracts` 7 families + re-keyed completeness | T6 `test_completeness_is_keyed_per_exported_constant_planted`; T18 Step 3; `tests/test_metric_contracts.py` |
| 9.4 `SCALE_GUARDED` growth guards + regressed-rescan proof | T18 Step 8 |
| 9.4 FFT/phasor calls constant in K | T13 `test_fft_and_phasor_calls_constant_in_k` |
| 9.4 numba cache key | T12; `tests/test_ci_shard_wiring.py::test_numba_cache_key_covers_all_njit_files` |
| 9.4 F1b upcast-gate `_KERNEL_DIRS` at the rebase | T26 Step 3 |
| 9.5 liveness on committed fixtures + fixture precondition | T18 Steps 5–7 (R2) |
| 9.5 `@slow` placement + `.test_durations` regenerated | T18 Step 7; T28 Step 5 |
| 9.5 driver tests: reducers, codegen reproduction, resume, exclusion, pre-flight, side-effect paths | T19–T23 |
| 9.5 both pandas legs + CI-scope lint before any commit | T25 Step 3, T28 Step 6 |
| §8.2 D1 procedures | T20 reducer tests (incl. `welch_segment_rule` R4 pin) |
| §8.4 D2 (levels, missing shard fatal, resume identity, gate, fallback) | T22 tests |
| §8.5 H1–H7 both sides of every threshold | T19 hypothesis reducer tests; T23 reduce tests |

---

## Self-review record (plan author, 2026-09-26)

- **Spec coverage.** Every section maps to a task:
  - §7.1 package and seams → T2–T5, T7–T18;
  - §7.2 surface → T17/T18;
  - §7.3 → T16;
  - §7.4 → T16;
  - §7.5 → T2/T3;
  - §7.6 → T15/T16;
  - §7.7 → T16;
  - §7.8 → T8–T12, T17;
  - §7.9 → T13/T17;
  - §7.10 → T16/T17/T18 (C24);
  - §7.11 → T16;
  - §7.12 → T14;
  - §7.13 → T14/T17;
  - §7.14 → T14;
  - §7.15 → T2/T12/T13/T18/T24;
  - §7.16 → T0/T3/T4/T7/T21/T26;
  - §8 → T19–T23, T27;
  - §9 → the mapping above;
  - §10 → T12/T21/T25;
  - §11 → T18/T24/T28;
  - §12 → T24/T28;
  - §13 → T25–T28.

  No spec requirement is dropped or deferred. The five spec gaps and errata found at plan time are owner rulings
  R1–R5 or concretisations C1–C27, each named where it applies.
- **Placeholder scan.** `ADR-NNN` and `PR-S` numbers are deliberate commit-prep assignments (§16), never TBDs. No step
  says "add appropriate …" without the content.
- **Type consistency.** Names are checked across tasks:
  - `pack_groups` → `(pos, counts, first_row)` (T2/T3);
  - `CoordinationSignals` / `PeriodSignals` fields (T16 → T17);
  - `COORD_*` constants (T14 → T17/T18);
  - `render_generated_params` (T19 → T20/T22/T28);
  - `objective_id` (T21 → T22 uses the C27 id instead, by design).

## Review history

- **Plan review r1** (`D:\Development\_reviews\2026-09-26-tf58-team-coordination-plan.md`, sha `96413c3c…`): APPROVE WITH
  FOLLOW-UPS, no BLOCKING; the reviewer concurs with R1–R5. TF58-PLAN-01 (CONSIDER): the Task 28 Step 5 throwaway-branch
  commit contradicted the Global Constraints "never commit/push" wording. Fixed:
  - the Commits bullet now names the two approval-gated exceptions, (a) the Task 26 force-push and (b) the Task 28
    `.test_durations` capture, neither of which adds a commit to the feature branch;
  - Task 28 Step 5 is rewritten as one approval covering branch, commit, push, draft PR and cleanup.

  Verifying the finding exposed a real gap. `ci.yml:3-7` runs CI only on `pull_request` to `main` or a push to `main`,
  so the original bare branch push would never have run the capture job. The draft PR fixes that.
- **Post-review update:** ruthless 0.7.0 is released, so the prerequisite and the Task 21 precondition now install from
  PyPI.
- **Owner rulings, 2026-09-26:** R1–R5 approved, with three strengthening amendments folded in for the r2 re-review.
  - **A1** — `computed_nonconverged` joins `COORD_SURROGATE_SOURCE_VALUES`, and the per-row surrogate-source precedence
    becomes disabled → not_scored → segment_too_short → computed_nonconverged → computed. Touches: R1 text, Task 14
    vocabulary, the Task 17 rule and test.
  - **A2** — an adaptive exact orientation predicate (Shewchuk 1997: a float filter with `ccwerrboundA`, plus a
    `fractions.Fraction` fallback on unsettled rows) replaces `det == 0.0` in the hull kernel. Touches: R3 text, Task 2
    code, 4 new tests including the float-false-positive non-vacuity case, the `NOTICE` citation, the CHANGELOG
    sentence.
  - **A3** — Tier-B base defaults are single-sourced in the generated file (`BASE_COORDINATION_PARAMS`,
    `BASE_SOURCE`). Commit 1 renders `INTERIM_BASE` (the R4 table). D1 derives pooled corpus-wide values. D2 applies its
    moved selections to them too. Task 28's `test_base_defaults_are_derived` blocks an interim base from a release.
    Touches: R4 text, Tasks 14, 19, 20, 22, 28.

    As written in r2, A3's pooled distributions were **match-weighted**, and they are superseded by the r2 fix below.
- **Plan review r2** (`D:\Development\_reviews\2026-09-26-tf58-team-coordination-plan-r2.md`, sha `980948f2…`): APPROVE
  WITH FOLLOW-UPS.
  - TF58-PLAN-01 is RESOLVED. R1–R5 and A1–A3 are ledger-closed.
  - **TF58-PLAN-02** (SHOULD FIX) is **fixed**. Match-weighting removed the within-match run-volume confound but left
    the 909 : 71 (92.8%) provider skew, so the r2 rationale "stops SkillCorner dominating" was wrong. The plan now
    states the base's intended population, "a provider not in the map" (Task 20 Reduce), and pools
    **provider-neutrally**: each provider counts once, and each match within a provider counts once.
    - `unit_weights` and `weighted_quantile` (numpy `inverted_cdf`) serve both the per-provider and the pooled values.
    - The match-weighted corpus-representative value is kept as a recorded diagnostic.
    - The tests assert provider-neutral equality with non-vacuity in both directions (match-weighted ≠ run-weighted ≠
      provider-neutral).
  - **Outside this round (acted on):** a `<0.7.0` ceiling from the F1b cycle's CI hotfix may be on `main`. Task 21
    Step 4, Task 26 Step 2 and C17 now remove any ceiling in the same commit as the complete `objective_id` migration,
    verified by `git grep`. `origin/main` checked 2026-09-26: no ceiling yet.
- **Plan review r3** (`D:\Development\_reviews\2026-09-26-tf58-team-coordination-plan-r3.md`, sha `b98edd96…`): APPROVE.
  - TF58-PLAN-02 is RESOLVED: provider-neutral pooling, with the intended population stated.
  - **TF58-PLAN-03** (CONSIDER: a thin provider carries the same 1/P weight as SkillCorner) was **taken, owner-approved
    2026-09-26**:
    - equal provider weighting is kept as a deliberate, stated choice;
    - there is no minimum-match floor, since any floor would be arbitrary;
    - D1 adds `provider_bootstrap_se` (match-level, 1000 resamples, seeded) and `thin_provider_flags` (flagged when a
      provider's SE exceeds the between-provider SD), recorded in `derivation.json`;
    - flags are reported only, never dropped or re-weighted;
    - Task 28 Step 1 stops for an owner ruling on any flag;
    - two-sided tests: the planted noisy thin provider is flagged, the well-sampled ones are not, and weights are
      unchanged.
