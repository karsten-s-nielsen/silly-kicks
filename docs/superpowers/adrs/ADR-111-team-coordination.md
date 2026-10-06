# ADR-111: Team-coordination dynamics (`silly_kicks.coordination`)

| Field | Value |
|---|---|
| **Date** | 2026-09-28 |
| **Status** | Proposed |
| **Deciders** | Karsten Nielsen (owner); design by Opus 5.5; implementation on the owner's chosen model |

> **Draft — design section only.** Per the plan (Task 24), this ADR is written at commit 1 with the design,
> decisions, concretizations and the pre-launch cost measurement. The **results section** (real-corpus D1 derivation,
> D2 gate outcome, D3 hypothesis verdicts and reliability) is written at commit 2, after the owner's DGX run.
> Depends on ADR-106 (F1b float32 frames), ADR-107/108 (native DAS), ADR-109 (detection primitive).

## Context

silly_kicks values individual actions (VAEP, xT) and has a rich tracking-feature layer, but nothing measures
**team coordination dynamics** — how the two teams' collective shapes couple over time. The football-science
literature (Bourbousson 2010, Moura 2012/2013/2016, Duarte 2013, Folgado 2014, Richardson 2012) defines these as
relative phase, cross-correlation, vector coding, spectral median frequency, coherence, cluster-phase synchrony
and the relative stretch index over team centroids and spreads. These are tracking-only, continuous-time signal
measures — a different kind of quantity from the action-coupled features already shipped, and they need their own
signal-preparation, windowing, surrogate-baseline and orientation machinery.

The forcing constraints: the measures must be **pure** (pandas in, pandas out; hexagonal), run on the ~980-match
TF-58 tracking corpus (SkillCorner 909 broadcast-tracking + GradientSports 64 + IDSSE 7) within the corpus cost
bound of §5 goal 6 (as ratified 2026-09-29: the per-match cost is provider-scale-dependent and the corpus bound,
≤ 1 h on 16 disjoint-slice workers, binds; §7.15's original ≤ 45 / ≤ 15 s per match is superseded), survive
SkillCorner's partial detection, and be
**calibrated and validated in-cycle** against the papers' reported effects before any default ships. Parameters
must be reproducible and provenance-stamped, never hand-tuned to fit the budget or the hypotheses.

## Decision

Add a new `silly_kicks.coordination` package (Architecture approach A, D10) implementing the seven method
families over a signal-agnostic three-level model (team–team, sub-unit, dyad), with **three parameter tiers**
(A paper-fixed, B derived from the corpus, C gated sensitivity) and **three owner-run construct-validity
drivers** — D1 `derive_coordination_params` (Tier B), D2 `calibrate_coordination` (Tier C, gated) and D3
`validate_team_coordination` (the in-cycle validation artifact, reported not gated). The library ships raw
primitives only; composites/archetypes/rankings stay consumer-side (ADR-009). No default changes without a D2
gate clearing on the corpus.

## Decisions (owner, 2026-09-26 — spec §4, D1–D21)

The full text of each is in **spec §4** (`docs/superpowers/specs/2026-09-26-tf58-team-coordination-design.md`).
As ruled:

- **D1** three analysis levels through one signal-agnostic pipeline; **D2** cluster-phase in; **D3** windows from a
  `windows` port (period / possession-from-events / possession-from-frames); **D4** segments split only at dead-ball
  stoppages > `max_stoppage_s = 25 s`; **D5** three parameter tiers (A/B/C), no AlphaEvolve; **D6** surrogate
  baselines on every coupling metric (circular time-shift default, K = 199; IAAFT opt-in); **D7** SampEn /
  Cross-SampEn (m = 1, r = 0.2·SD, O(N log N)); **D8** per-method paper-faithful GK policy, no splicing; **D9**
  detection-aware SkillCorner handling + a synthetic broadcast-occlusion leg; **D10** new package + seam
  improvements; **D11** design §§1–5 as presented.
- **D12 (R1)** vectorised convex-hull area replaces Qhull (`convex_hull_area` within 1e-9); **D13 (R2)**
  `compute_defensive_line` vectorised, byte-identical; **D14 (R4v2)** the sweep is cached per-level corpus passes +
  a ruthless-efficiency grid; **D15 (R5)** phase on maximal continuous segments, windows only select samples;
  **D16 (R6)** optional `numba` kernels with numpy reference + parity gate; **D17 (R7)** dyads default to period
  windows; **D18 (R3/R8)** phasor representation for relative-phase stats + structural cost gates.
- **D19** forward-compat with F1b + native DAS; **D20** dead-ball evidence via a fail-closed provider taxonomy
  (`resolve_stoppages`, precedence ball_state → events → unavailable); **D21** ruthless 0.7.0 (`GridSearchStrategy`,
  required `StoreConfig.objective_id`).

### Owner rulings R1–R5 (plan) with amendments A1–A3

- **R1** vectorised hull; **R2** vectorised byte-identical defensive line; **R3** phasor stats; **R4** cached
  per-level passes + ruthless grid; **R5** unit-vocabulary amendment (below, ADR-048).
- **A1** the surrogate token `computed_nonconverged` (a non-converged surrogate draw is recorded, not silently
  dropped). **A2** the exact orientation predicate (Shewchuk 1997, adaptive-exact; cited in `NOTICE` beside the
  TF-58 paragraph) in the hull kernel. **A3** the Tier-B base is single-sourced in the generated file
  (`_provider_params_generated.py`), `BASE_SOURCE` moving `interim` → `derivation` at commit 2.
- **TF58-PLAN-02** the pooled base is PROVIDER-NEUTRAL (one draw per provider, match-weighted within), so it never
  inherits SkillCorner's 909/980 signature. **TF58-PLAN-03** equal weighting is kept; D1 additionally reports a
  thin-provider bootstrap SE + flags (report-only; Task 28 stops for the owner on any flag).

## Concretizations that change behaviour (C1–C27; the ones that alter an output or a contract)

- **C1** `compute_defensive_line` byte-identity rests on count-bucketed compact rows: each bucket's rows reduce with
  numpy's 1-D routines exactly as the per-group loop did, so the pairwise-sum order AND the DEFAULT argsort's tie
  order (non-stable, SIMD-dispatched -- the legacy loop used it too) are identical to the loop on every platform;
  the fixtures carry exact back-line ties.
- **C3** the stale-artifact detector covers D1/D2/**D3** outputs — `_artifacts_for` scans `rglob("*.json")`.
- **C4** `objective_id` on a dirty tree carries a `+dirty-<uuid4>` suffix so a dev run's store can never collide
  with a clean run's.
- **C12** two keyword-only overrides on `build_coordination_signals` (`stoppage_evidence`, `detection`) — needed by
  the D2/D3 legs; the public positional signature is unchanged.
- **C13** a `dyad` row has no vector-coding or coherence method (`coord_vc_*`/`coord_coh_*` are NaN) — those are
  team-level constructs.
- **C14 / TF58-IMPL-06** `build_selection_artifact` (`calibration/_selection.py`) is carrier-specific (hard-codes
  `beta`/`gamma`), so D2 does NOT reuse it; a generic `_selection_dict` serialises the coordination `Selection`.
  (Spec §8.4 amended to record the deviation; the re-implementation is faithful to ADR-060.)
- **C17** three ruthless pins bumped to `>=0.7.0` in `pyproject.toml` (the F1b `<0.7.0` CI cap is lifted in the same
  commit, C17/§13 item 1) and every `StoreConfig`/builder site migrated.
- **Corpus de-risk findings (MEDIA-PC local no-flip subset, 2026-10-01) -- two defects fixed, TDD, both sides
  tested.** (1) `possession_windows_from_actions` raised `KeyError` on a match with actions in a period the frames do
  not cover -- a per-match data hole, not a provider property: GS 10510 and 10511 ship events for periods 1-4 but
  tracking for 1-2 only, while the other GS extra-time matches (10506, 10508, 10517) ship full period 1-4 tracking.
  Such actions now get no window (nothing to score, no period end to close it) and a `CoordinationCoverageWarning`
  names the skipped (game, period)s; the (game, period) keys are canonical (`id_compat`) because an `iloc` row upcasts
  an int `game_id` to float, and the two raw id `==` in the builder now use `same_id`/`ids_match` (ADR-019).
  (2) `_xc_identity` raised "slice lies in 0 B segments (a bug)" on GS 3828: an on-pitch-count change splits a
  segment with NO gap (`_segments_with_count`, "red card splits"), and `_both_runs`' union masks let a slice run
  across the cut -- correctly for the direct null (each part rolls by its own draw), but one circular FFT per slice
  cannot express two independent draws. The first fix (2026-10-01) made the identity decline that slice. The
  independent reviews (2026-10-02, A's M1 / B's M-13) showed the root, which this ADR first called an "open method
  question": spec 7.4 step 2 + D15 define every slice as window ∩ A-segment ∩ B-segment, so the OBSERVED XC / VC /
  coherence / RSI statistics running across the cut were a spec deviation, not a choice. Fixed: `_both_runs` is the
  per-segment intersection sweep (touching segments split), every family's observed statistic and null read it (RSI
  sign switches included), and `_xc_identity`'s strict raise is restored (a slice can no longer touch two B
  segments) -- a corrected-output change (item 3 below). The brief on-pitch-count excursions an inexact substitution
  handover leaves were MEASURED first (owner ruling 2026-10-02), on the 50-match subset: on GradientSports every
  match has them (95 excursions; 53 return to the prior count within 22.5 s, 37 within 2 s -- typically the incoming
  player appears ~1.1 s before the outgoing one leaves; the next lasts 42.5 s, 26 last over a minute, 15 never
  return: sendings-off or a player's data ending near the whistle); SkillCorner and IDSSE count changes all sit at a
  detection gap or inside a long stoppage, so they add no split of their own. **Owner ruling 2026-10-03:** a count
  excursion that returns to its prior count within `max_stoppage_s` (25 s; no new parameter) is a handover and does
  not split (`_signals._absorb_handovers`; samples keep their values); a longer or non-returning change splits.
  Spec 7.4 step 2 amended. Two crash-poisoned cache entries (GS 3852, IDSSE J03WOY; zero-filled at the 2026-09-29
  crash) were purged and re-downloaded.
- **Cluster minimum window (owner ruling 2026-10-02) -- a corrected-output change, like F7.** The subset no-flip
  verdict after the two fixes above was 50/50 matches, 0 source flips, 0 NaN changes, but 12 percentile crossings,
  ALL in `cluster_team.coord_rho_group_mean_percentile` and ALL on single-sample 0.1 s possession windows (IDSSE
  J03WOH/J03WPY/J03WR9): with one usable sample each player's window-mean relative phasor is its one relative
  phasor, so `rho_group` is exactly 1 for the data and every shift draw, and the percentile is decided by float
  rounding (the reference leg alone ranged 0.09-0.99). So D4 changed no real conclusion; the cluster family lacked a
  minimum. Ruled: a window with fewer than `MIN_CLUSTER_SAMPLES = 2` usable samples is `too_short` (rho mean/sd,
  SampEn NaN; surrogate `not_scored`; no player rows); spec 7.8.6 amended. Only single-sample windows change.
- **C17 x main's study-parallel trainers (merge 2026-10-01).** `main` (79e0f0f) split the xShot/xCross nested HPO into
  `run_one_study` workers + an `assemble_studies` reduce sharing a per-study `<tag>.study.json` cache. The D21 identity
  is carried through that path: the prep persists `objective_inputs` in the study config (orchestration knobs
  `shard_root`/`study`/`assemble` excluded like `output_dir`/`n_trials`/`allow_dirty`), workers and the reduce key
  their stores on it + the persisted `run_prov`, and the shard records the `objective_id` + `n_trials` it was computed
  under and is served ONLY on a match (`scripts/_study_shared.read_study_shard`/`write_study_shard`) — the store's own
  resume rule. Without that, a shard left by another run (other code / corpus / `--n-trials`) was served stale and a
  dirty tree resumed across code it cannot describe (C4). Consequence: a `--allow-dirty` parallel run recomputes its
  studies in the reduce (fail-closed; clean runs resume fully). That recompute is real: a dirty id opens a FRESH
  sibling store (`scripts/_provenance.store_path_for`, review M-6) -- reopening the worker's store under the reduce's
  new nonce would make ruthless refuse ("written for a different objective"); the same rule serves every D21 site
  (both trainers, `calibrate_tracking_defaults`, `calibrate_xt_bandwidth`) and D2's C27 store (`dirty_suffix`).
  Guarded by `test_train_study_shard_identity.py` (incl. a real-strategy dirty-tree test) and a non-vacuity assert in
  the byte-identity parity tests (the reduce served every worker's shard).
- **C19** spec §9.1's 225° vector-coding example is a documented regression pin (the printed Moura Eq. 2 abs-value
  misclassifies a 225° coupling; the kernel is correct, the example is the wrong angle).
- **C20** team sync uses the segment-level cluster amplitude (φ̄_k over the continuous segment, not the window).
- **C22 / C8** the team-pair tables (`team_sync`, `rsi`) carry `team_a_id` + `team_b_id`; all id columns follow the
  nullable-dtype + `id_compat` rules (ADR-019/058).
- **C24** orientation is applied algebraically per window from team A's goal-relative frame (a 180° point
  reflection), which keeps "phases once per segment" (D15) compatible with per-window orientation.
- **C25** the window-source-mix rule: events and tracking are never mixed in one `windows` call (D3 refusal).
- **C26** D2 levels are relative to each provider's Tier-B value (multipliers, or additive offsets for band edges).
- **C27** the D2 `objective_id` is a digest spanning every shard generation it reads (so a changed upstream shard
  invalidates the store).

### Implementation concretizations added this cycle (record; faithful to spec, not pinned by it)

- **D1 apply_filter seam (owner-ruled Option 1).** `_build_coordination_signals(..., apply_filter=bool)` exposes the
  raw-resampled-aligned signals D1 Pass B needs for `vc_epsilon`; the public `build_coordination_signals` is a thin
  `apply_filter=True` delegator (byte-identical). `_result_from_signals` is the signals→result seam D1/D3 reuse.
- **D1 occlusion is spec-based (owner ruling).** Occlusion bites via extrapolated positions
  (`simulate_broadcast_occlusion`); both full and occluded are scored fully-observed; the observed fraction is
  computed EXTERNALLY from `fov_mask` per window — NOT `detection="detection_aware"` (a no-op for the fully-observed
  GS/IDSSE).
- **The Layer-2 discarded-visibility pre-flight (spec 8.3, ADR-069; owner ruling 2026-10-03).** Spec 8.3 names
  `validate_corpus_visibility`, which reads parquet shard metadata; the TF-58 drivers never read parquet shards (they
  load each match from raw provider files through the pining loader). A per-match check alone would repeat the very
  failure ADR-069 was written for -- a discarded flag surfacing deep in the pass, after every other provider's
  matches had been computed. So every corpus pass first runs `_coordination_corpus.visibility_preflight`: one match
  of each detection-aware provider in the slice, loaded through the pass's own loader, must carry `visibility`, or
  the pass refuses with the ADR-069 remedy before any compute (a match that fails to load for another reason is
  skipped for the next, up to three). The per-match `detected_mask` all-null trap stays for a single match's hole:
  `for_each` records it failed and every combine refuses the key (B-1). A derived registry gate holds every pass
  that opens and walks the corpus to the pre-flight; behaviour tested end to end on the D3 metrics pass. The pre-flight
  is a loading loop outside `for_each`, so ADR-052's Rule C lists it as an `_UNSHARDED_LOOP_EXEMPT` home, with its
  reason: a bounded probe (at most `attempts` loads per detection-aware provider, one at a time, nothing kept), never a
  corpus pass. A test pins that bound.
- **D2 joint-preparation pass (TF58-IMPL-04).** A confirm point moving ≥ 2 parameters (any prep/post-prep mix) runs
  a real joint-preparation corpus pass at the joint's preparation values -- its own partitioned layer, `--layer
  joint`, which reads the joint from `oat.json` (layer b) -- not a read of the wrong precomputed file; confirm
  refuses shares computed for another joint. Switch-events are single-sourced (`rsi_switch_times` +
  `possession_change_times` emitted by `match_tables(include_switch_events=True)`), so H7 is live at D2-confirm and
  D3 (TF58-IMPL-05).
- **D3 reliability entity.** `team_id` for team-keyed tables (spectral, cluster); the MUTUAL team-pair tables (pair
  `team_team`, `team_sync`, `rsi`) are melted onto BOTH participating teams (`team_a_id`/`team_b_id`).
- **D3 coverage stratification.** A per-window observed fraction, derived from the tables that carry one (pair
  `coord_observed_fraction_a`, cluster `coord_observed_fraction`), is joined to EVERY representative metric, so
  spectral/team_sync/rsi (no per-row observed fraction) are still stratifiable by observed-fraction decile.
- **D3 Type-II slope** is the major-axis slope of the two match-halves' per-team means (same `_stable_half` split as
  `split_half_reliability`, `scripts/_reliability.py`).
- **D3 reads the calibrated occlusion width W from `derivation.json`.** D1's `build_derivation` records
  `occlusion.width_m` -- the ONE width every occlusion worker used (calibrated in `--pass occlusion-cal` from every
  worker's histogram; the workers' manifests must agree) -- so D3 runs its occlusion leg at the SAME W without
  re-calibrating. Additive and byte-safe for the codegen (`render_generated_params` reads only
  `derivation["pooled"]`/`["providers"]`).
- **Partitioned corpus passes + the artifact handoff (review B-1 / M-3 / M-4; owner ruling M-5, 2026-10-02).** Each
  worker writes only its share of a pass (`<pass>.<tag>.parquet` + a manifest naming every key it was handed and what
  became of it); every corpus-wide step -- D1 pass b / occlusion / reduce, D2 layer b / confirm, D3 reduce, the
  numerics reduce -- combines ALL shares first (`_coordination_corpus.combine_workers`) and refuses a missing worker,
  a failed key, a key handed to two workers, mixed generations or commits, workers that disagree on a declared field
  (a calibrated width, an artifact digest, a joint point), or a population other than `--corpus-json` (spec 8.3:
  never a partial corpus). The combined table is sorted stably by (provider, match), so it does not depend on the
  split. Shard tokens carry the VALUES the work consumes (M-4), digested over EVERY TF-58 provider
  (`run_params_token`): workers are launched with their own slice's `--providers`, and a digest over those alone split
  one run into per-worker shard generations -- the fresh 50-match no-flip reduce of 2026-10-03 refused exactly that,
  and a two-worker test now pins it. D1's reduce writes `derivation.json`
  and the generated module into its `--out`; D2 and D3 compute with that artifact (`--derivation`, and D3 also
  `--calibration`) through `params_from_artifacts` -- exactly the maps the codegen renders -- with its sha256 in every
  token, manifest and artifact; D2's confirm writes `calibration.json` and the final module into its `--out`; the
  package is never rewritten on the DGX, and commit 2 copies the artifacts in. D2's C27 id hashes, per level read,
  `<level>:<shard generation>:<population digest>` (+ the C4 dirty nonce), so a regenerated level or a grown
  population never resumes stale scores. Every artifact (`derivation.json`, `calibration.json`, `metrics.json`,
  `numerics_noflip.json`) records `corpus_visibility` (ADR-038, spec 8.3; review minor 13): `artifact_label` over the
  (provider, match) pairs its aggregates came from, keyed on the pining manifest's per-match `visibility` (never the
  provider name; an unlisted match is private) behind `assert_public_corpus`
  (`_coordination_corpus.corpus_visibility_label`).

## Consequences

### Positive

- A first team-coordination layer, calibrated and validated on the corpus against the papers' reported effects
  before any default ships; raw primitives compose into VAEP/consumer analytics.
- Reusable seam improvements: vectorised hull + defensive line (byte-identical), the `windows`/`resolve_stoppages`
  ports, the shared driver discipline (`_coordination_corpus`), and the `_reliability` kernels shared with TF-52/62.

### Negative

- A large new surface (7 families, 3 drivers, ~980-match passes) with a real per-match cost; the corpus passes are
  owner-run on the DGX (not CI). SkillCorner partial detection means several measures degrade honestly to NaN, and
  the interim R4 base makes a few spectral/coherence columns unreachable on real football until D1 lands (liveness
  uses synthetic well-posed regimes until then, real-data liveness deferred to D3).
- ruthless 0.7.0 is a hard dependency bump (the F1b `<0.7.0` cap is lifted in the same commit); every StoreConfig
  site migrates together.

### Neutral

- The Tier-B numbers live only in the generated `_provider_params_generated.py` (A3); `BASE_SOURCE` records
  `interim` vs `derivation`. The three drivers refuse a dirty tree (ADR-037) and stamp provenance.

## Performance

- **Budget.** §7.15's original per-match budget (≤ 45 s/match numpy-only, ≤ 15 s/match with `numba`, on 10 Hz
  analysis) is SUPERSEDED by §5 goal 6 as ratified 2026-09-29: the per-match cost is provider-scale-dependent and the
  corpus bound binds (below). The table's last column shows the original budget for reference only.
- **Seam benchmarks (plan Task 0 Step 4 vs Task 3 Step 8).** `compute_team_shape` (per frame per team) and
  `compute_defensive_line` (per frame-team) on `tests/datasets/elastic_sync/j03wmx_slice/frames.parquet` (6,201
  frames, 2 teams), best of 5 per run, two interleaved runs, the pre-change code exported from `main` (`ba9c151`,
  `git archive`) against this tree, same interpreter (py 3.14.2, pandas 2.3.3, this laptop, 2026-10-03):

  | Seam (µs per frame-team) | Before, runs 1 / 2 | After, runs 1 / 2 | Speed-up |
  |---|---|---|---|
  | `compute_team_shape` | 1,244 / 1,135 | 197 / 191 | about 6x |
  | `compute_defensive_line` | 92.8 / 84.8 | 6.0 / 9.3 | about 9-15x |

  (Spec §3.2's investigation probe measured 1,256 / 152 µs before the change on py 3.14 + pandas 3; the defensive-line
  difference is the pandas major.) The outputs are byte-identical except the hull area (D12, within 1e-9) -- gated by
  the collective parity tests (`tests/tracking/test_collective_parity.py`). The Task 0 Step 4 baseline was not
  recorded when Task 0 ran; it was measured here from the exported pre-change code (review M-11, 2026-10-02).
- **Pre-launch real-match measurement (plan Task 24 Step 1), one match per provider, D3 `--pass metrics`
  (`n_surrogates = 199`):**

  | Provider | Rate | numba s/match | numpy s/match | Original §7.15 budget (superseded) |
  |---|---|---|---|---|
  | skillcorner | 10 Hz | 18.1 (k = 0: 13.7) | 25.2 | ≤ 15 / ≤ 45 |
  | idsse | 25 Hz | 113.8 (k = 0: 32.0) | 437.2 | ≤ 15 / ≤ 45 |
  | gradientsports | 29.97 Hz | 131.3 (k = 0: 34.5) | 453.5 | ≤ 15 / ≤ 45 |

  _(2026-09-29, after every decision below incl. the owner's rulings A / B / C on the second measurement; per-stage
  breakdown in the D3 manifest's `stage_seconds`. History, numba: 20.0 / 705 / 1,034 s before the decisions; 18.8 /
  127.0 / 184.9 s after D1-D5 (numpy 22.1 / 468.0 / 468.4; k = 0 16.9 / 40.7 / 40.0). F7 restored work the defect
  had skipped, so SkillCorner's cluster family rose 0.7 -> 3.9 s numba while its signal prep fell 8.6 -> 5.2 s.)_

  **Corpus (the binding bound per ruling C):** SkillCorner 909 x 18.1 s + GradientSports 64 x 131.3 s + IDSSE 7 x
  113.8 s = 7.1 h single-core, **27 min on 16 workers** with numba (goal 6's "<= 1 h on 16 disjoint-slice workers":
  met; the §7.15 cost model's 3-12 h: inside). numpy-only: 15.3 h single-core, 57 min on 16 workers.
  _Goal 6 restated per ruling C (provider-scale-dependent per-match cost; the corpus bound binding) -- RATIFIED by the
  owner 2026-09-29 and written into the spec (§5 goal 6; §7.15 "Measured"), resting on the per-provider per-match
  floors above and the per-stage timers in every D3 manifest (R8)._ Memory, measured during the local no-flip trial:
  a worker loading one full-tracking match holds ~7-8 GB resident (the §7.15 model said ~2 GB), so the DGX run sizes
  its worker count by memory as well as cores.

  _A miss is reported to the owner with the stage breakdown before any further step; no method is trimmed to fit the
  budget without an owner ruling._

### Performance decisions (owner rulings 2026-09-28)

The first real-match measurement missed §7.15 by 1-2 orders of magnitude on the full-tracking providers. The owner
ruled on five decisions after an independent review; each change below is either byte-identical (and gated so) or a
pre-authorised numerics change behind a hard no-flip gate. No method, draw count or analysis rate was changed.

- **D1 -- cluster surrogate shift unit = each player's own phase runs (plan R1), a correctness fix.** The as-built
  null shifted the team's segments, so a window with no overlapping team segment returned its observation K times (a
  zero-variance "computed" null; SkillCorner 46 rows) and rolled NaN phasors into valid slots. Hard gate: 0 computed
  rows with a zero-variance null on the three real matches (met: SkillCorner 46 -> 0 computed, all honest
  `segment_too_short`; IDSSE 1,052 -> 1,576 computed; GradientSports 1,744 -> 1,756), plus a mandatory non-vacuity
  test (players locked to one irregular rhythm put the observation at the top of its null). The null distribution
  changes by design. (The shift UNIT stays each player's own phase runs; review B m3 later refines WHICH runs are
  checked -- only the contributing ones, see item-3 "Surrogates draw from the contributing segments" -- so these
  computed / `segment_too_short` counts move again and are re-measured at the commit-1 reference regeneration.)
- **D2(a) -- byte-identical batching of the four pair nulls.** Series B is gathered for all draws at once
  (`shifted_window`, never a per-draw full-period copy + `np.roll`), each family's metric computed on a leading draw
  axis with every reduction on a C-contiguous array along the reference's axis. Held bit-for-bit against the per-draw
  legacy loop (ADR-105 oracle pattern) and on the real matches (0 mismatches). Finding recorded: **scipy 1.18's
  batched FFT is not per-row identical** to the 1-D transform (measured on the CI-mirroring venv), so the
  cross-correlation and coherence direct nulls call the 1-D scipy kernels per draw row -- identical by construction on
  every scipy version -- and batch only what is version-proof (A's spectrum once, the gathers).
- **D2(b) -- the spec 7.9 identities, wired (spec conformance; the direct nulls were an unrecorded deviation).**
  Relative-phase R per B segment by ONE circular cross-correlation of A (zero outside the idx rows) with the segment;
  % near-in-phase by direct counts (`near_in_phase_counts_at`, numba optional, identical counts in both backends);
  cross-correlation by the circular FFT cross-correlation over the segment, the exact correction of the |lag| terms
  that fall outside the slice (ascending sums, numba optional, bit-identical backends) and circular prefix sums
  (`shifted_slice_lagged_pearson`, generalised from whole segments to any slice). The identities decline -- the
  direct null runs -- where a touched B segment holds a non-finite value (the FFT would spread it to every shift).
  Parity, measured: R <= 4e-17 (kernel), 8.9e-16 (IDSSE, 4,366 rows); near-in-phase counts identical; cross-correlation
  max |r| 1.3e-14 (IDSSE, 968 rows) / 3.8e-14 (GradientSports); on SkillCorner every pair null is
  `segment_too_short`, so the identities never score there.
- **No-flip on the three real matches (D2(b) + D4 together, k = 199, against the reference numerics):** 0 source-token
  flips, 0 percentile changes, 0 NaN-pattern changes (IDSSE: 4,366 RP / 968 XC / 1,576 cluster / 500 team-sync
  computed nulls; GradientSports: 6,964 / 968 / 1,756 / 870). The full-corpus run is the owner's DGX gate.
- **D3 -- Butterworth: cached steady-state `zi` + an exact `sosfiltfilt` replica** in the shared
  `tracking/preprocess/_butterworth.py` (odd extension, two public `sosfilt` passes seeded `zi * x0`), read-only
  cached arrays; a fence test holds the replica to scipy's real `sosfiltfilt` bit-for-bit over the run-length battery
  (passes on scipy 1.17.1 and 1.18.1).
- **D4 -- one redefinition of the cluster + team-sync reference arithmetic.** Explicit real arithmetic for every
  complex product, sequential sums in a fixed documented order (players in column order, rows ascending, from +0.0),
  unit phasors as `s / sqrt(re^2 + im^2)` with `1 + 0j` where `|s| == 0` (the value `exp(1j * angle(0))` gave), and
  the team-sync Pearson by an explicit two-pass formula instead of `np.corrcoef`. The numba null and the numpy
  reference then agree bit-for-bit (the as-built complex multiply takes a CPU-dispatched FMA path no numba kernel can
  follow), and the observed estimator and every surrogate draw share one arithmetic. Parity against the as-built
  arithmetic, measured: cluster null <= 7.8e-16, observed statistics <= 2.4e-15, team-sync Pearson 7.8e-16.
- **D5 -- scale-guard ladder (16, 32, 64) with `max_exponent=1.2`** on the signal-build guard AND its rescan
  companion (measured: real 1.000, shim 1.649).
- **R8 -- per-stage timers in the drivers' manifests** (`windows`, `signals`, `family.<name>`, `family.combine`,
  `melt`, `switch_events`), so the breakdown is reproducible from the official artifact.
- **Reference numerics + the corpus no-flip gate.** `SILLY_KICKS_COORDINATION_REFERENCE_NUMERICS=1` scores through
  the as-built numerics (the direct nulls; `_cluster_reference`, the as-built cluster/team-sync code kept verbatim):
  the parity oracle of the tests and the reference leg of `scripts/validate_coordination_numerics.py`, the owner's
  HARD gate for D2(b) + D4 -- on the full corpus, no source token may flip, no surrogate percentile may cross 0.025 /
  0.05 / 0.5 / 0.95 / 0.975 and no value may change between NaN and finite; any violation stops the rollout for an
  owner ruling. Production never sets the switch.

### Owner rulings on the second measurement (2026-09-28, late)

- **A -- the identities on every window; relative phase size-dispatched.** Cross-correlation scores every time-shift
  null through the general identity (any slice, not only whole segments). Relative phase runs its identity only where
  it is cheaper than the direct null, by a FIXED size rule: identity iff `sum_g n_g * ceil(log2 n_g) <= 17/4 * K *
  |idx|` over the B segments the window's rows touch (`RP_IDENTITY_CROSSOVER`, `_rp_identity_is_cheaper`). The
  constant comes from measured unit costs (three complex FFTs ~3 ns per `n log2 n` unit each; direct count ~1.5 ns
  and direct null ~40 ns per draw x row: `(40 - 1.5) / 9 = 4.28`). The rule reads sizes only, in exact rational
  arithmetic, so the branch is deterministic on every platform; both branches are parity-equal (the identity-vs-direct
  test), so the rule chooses speed, never a result. Tested from both sides of the boundary, with a non-vacuity test
  that the fixture takes both branches; the C11 FFT-count gate pins the dispatch to the identity. Conditional on the
  full-corpus no-flip: a flip reverts the identities to whole-segment windows and is investigated.
- **B -- F7: resampling lost run-edge samples to float rounding (correctness fix).** Signal preparation resampled each
  run twice: onto a run-local grid (`resample_uniform`, whose default `n_out = floor(span * fs) + 1` truncates when
  `span * fs` lands a hair below an integer, e.g. `(288.9 - 282.0) * 10 = 68.99999999999977`), then onto the period
  grid with NaN outside the run-local grid. Edge samples an ulp outside became NaN, and `_phasors_over_runs` (which
  needs an all-finite run) then dropped the WHOLE run's phase. It also interpolated twice where a run starts off the
  grid (25 / 29.97 Hz native rates), which spec 7.4 step 4 ("resample ... by linear interpolation within runs") does
  not describe. Fix: ONE linear interpolation of each filtered run at the period-grid times it covers, and ONE
  float-tolerant edge rule, `grid_span` + `GRID_TOLERANCE = 1e-9` samples in `tracking/preprocess/_butterworth.py`,
  shared by `resample_uniform` (run edges and the default `n_out`), `resample_frames` (grid length and the
  "latest row at or before the grid time" hold) and the coordination period grid. Measured on the three reference
  matches: player runs that lost their phase went from 3,503 of 4,384 (SkillCorner; 80% of run samples), 7 of 54 (IDSSE)
  and 3 of 51 (GradientSports) to 0. SkillCorner team signals are unchanged (at most 6e-12) and gain 317 valid samples.
  IDSSE / GradientSports team signals move where runs start off-grid: about 35% / 52% of samples, by at most
  2.2 / 3.6 mm (centroid), 1.5 / 5.6 cm (spread) and 0.25 / 0.45 m² (hull area). Seam callers (AGENTS.md sweep
  rule, grep evidence): `resample_uniform` <- `resample_frames` + tests; `resample_frames` <- tests only (no library
  or script caller); coordination no longer calls `resample_uniform` (it reads `grid_span` and interpolates the run
  slice directly, O(run) instead of O(period) per run); `butterworth_lowpass` is unchanged (callers `smooth_frames`,
  `scripts/derive_coordination_params.py`).
- **C -- the floor, option (a): method-neutral speedups first, then restate goal 6.** Every change below leaves every
  output table byte-identical (held against the post-A baseline on the three reference matches: 0 mismatching
  columns), each with a test that fails if the saving regresses:
  - *Signal preparation* filters and resamples each player's rows once. The team signals reuse the cluster roster's
    series of every player without goalkeeper rows (the same rows, so the same arrays). x and y share one run split
    and one two-row Butterworth pass (`butterworth_lowpass_rows`; scipy's `sosfilt` runs each row through the 1-D
    recursion, fence-tested bit for bit on scipy 1.17 and 1.18). The period build moves only the columns it reads,
    and the duplicate refusals read only their key columns.
  - *Analytic phase* replicates `scipy.signal.hilbert` for 1-D input step for step (`fft`, the exact x2 / zero of the
    two spectrum halves, `ifft`) and numpy's `reflect` pad by slicing. This removes the generic per-call overhead
    on ~12k short SkillCorner runs per match. The replica follows scipy >= 1.17's in-place form and the fence holds it
    there bit for bit (verified 1.17.1, 1.18.1). scipy <= 1.16 (verified 1.15.3, 1.16.0; CI's py3.10 leg) builds
    `Xf * h` with a complex `h`, and a complex multiply by `1 + 0j` can flip the sign of an exact zero, so there the
    fence asserts equality in value; the two forms differ only in the sign of exact zeros (measured: 9 of 33 battery
    cases, all at exact zeros, as in constant runs). The phase no longer depends on which form the installed scipy
    uses.
  - *Cluster null*: the phasors' real/imag split happens once per (period, team, axis) context, not once per
    window (`phasor_parts`, `shifted_rho_group_means(..., parts=)`). Sample entropy's 1-D pair count is an exact
    two-pointer sweep under numba (`pairs_within_sorted`): same predicate, same integer, ties included.
  - *Pair nulls*: each B segment's identity side is prepared once per compute call and shared by every window rolled
    within that segment (`_SegmentCache`): R's `conj(fft(B))`; cross-correlation's centred B, its `rfft` and its
    prefix sums. The window/segment overlap tests are vectorised (`_both_runs`; since 2026-10-03 the null's
    segment choice is `_segments_holding`, a `searchsorted` over the rows the statistic reads).

  One reported counter changes, as a fix: `n_runs_too_short` counts a dropped run ONCE (spec 7.4 step 3). It counted
  every outfield run 4x (x, y, player path, team path) and a goalkeeper's 2x. One seam defect was found and fixed on
  the way: `resample_frames` grouped by `(period, is_ball, player_id)`, so two games in one frame dropped a game's
  rows; the entity now includes `game_id`.

### Three kinds of change, kept apart (owner, 2026-09-29)

The performance work produced three kinds of change, and each is gated differently:

1. **Byte-identical speedups (ruling C).** No output moves; each is held to the pre-change tables on the three
   reference matches (0 mismatching columns) and carries a structural guard.
2. **Invisible numerics changes (D2(b), D4, ruling A's dispatch).** Outputs move at float-rounding level (measured
   maxima above), with one amplified case on the tested-tree 50-match no-flip verdict (review minor 3):
   `cluster_player.coord_phi_sd_deg` moved by up to 1.4788e-6 degrees. The circular SD, sqrt(-2 ln R), has derivative
   -1 / (R * SD), so where a player is locked to the team (R near 1, SD near 0) it magnifies a ~1e-16 change in R;
   every other column moved at most 4.04e-11. The hard gate is the corpus no-flip
   (`scripts/validate_coordination_numerics.py`): no source-token flip, no percentile crossing, no NaN-pattern change.
3. **Corrected-output changes -- NOT byte-identical, because the old values were wrong:**
   - **F7 (resampling).** Every player run now keeps its phase (SkillCorner had lost 80% of them), and runs that
     start off the 10 Hz grid are interpolated once, as spec 7.4 says. IDSSE / GradientSports team signals move by
     a few mm (centroid) to ~5 cm (spread) on 35% / 52% of samples; SkillCorner's move only at rounding level.
   - **Possession windows from events.** They ordered actions by `action_id` alone; they now use the robust
     `(game, period, time_seconds, action_id)` key (ADR-065 §3d; found by its completeness gate). The windows move
     only where `action_id` is not chronological (never for fresh converter output).
   - **`include_goalkeeper` honoured per method.** The `team_signals` and `dyad` flags were validated but ignored
     (both paths hard-coded outfield-only). They now decide whether the keeper is one of the team's n players (spec
     7.5) and whether keeper pairs are dyads. The defaults (False / False / True) reproduce the previous tables byte
     for byte on the three reference matches; the non-default values now take effect (non-vacuity tests: the keeper
     moves team A's centroid by more than 1 m on the fixture; dyad tables gain 10 same-team + 21 cross-team keeper
     pairs per axis).
   - **Cluster minimum window (`MIN_CLUSTER_SAMPLES = 2`, owner ruling 2026-10-02).** Only single-sample windows
     change: they become `too_short` (above).
   - **Possession windows skip an untracked period.** Actions in a (game, period) the frames do not cover get no
     window and a `CoordinationCoverageWarning` names the period (de-risk finding 1, above); before, they raised.
   - **Slices respect the segments (review M-13 / A-M1, 2026-10-02).** Every observed statistic and null reads
     window ∩ A-segment ∩ B-segment slices (spec 7.4 step 2, D15); windows that crossed an on-pitch-count change
     (red cards, inexact substitution handovers) move.
   - **Long stoppages split every player's run (2026-10-03, found while writing the plan-named tests).** Spec 7.4
     step 2 splits a player's on-pitch time at dead-ball stoppages longer than `max_stoppage_s`, before the filter;
     the build split only the team segments, so every player phase (cluster phase, dyads) and the player filter ran
     straight through a long stoppage. Now one source (`_period_stoppages`) splits both: the rows inside a long
     stoppage are dropped before the run split, so the filter never smooths across it. Every match with a stoppage
     > 25 s moves in its cluster, dyad and team-sync rows (and, through the filter edges, slightly in its team
     signals near the stoppage). A dyad null now meets the same short pieces of play between two long stoppages that
     a team null always met: on the R2 real half the 100.4 s piece is long enough for cross-correlation to read but
     too short to shift at the interim `min_shift_s` = 60 s, so none of its cross-correlation nulls is computable
     (they were, through the dyads, before). Its three null columns join that fixture's measured limitations
     (`test_liveness`); the D1-derived `min_shift_s` decides how often this bites on the corpus.
   - **Surrogates draw from the contributing segments (plan Task 17; found 2026-10-03 while tracing the above).**
     The pair families and team sync refused a row as `segment_too_short` when ANY B segment overlapping the window
     was too short to shift -- including segments holding no row the statistic reads (B's run inside A's detection
     gap; a slice cross-correlation drops as shorter than its 60 s minimum), which the null never reads. Plan Task
     17's rule is "any contributing segment". Now each family hands `_surrogate` the slices its statistic reads and
     the null shifts and checks only the B segments holding them (`_segments_holding`; team sync: the rows finite on
     both sides); the IAAFT non-convergence count follows the same rule. **The cluster family follows the SAME
     "contributing" rule (review B m3, owner-ratified 2026-10-04): `_cluster_surrogate` shifts and checks only a
     player's runs that hold a usable `rho_group` sample the statistic reads (via `_segments_holding`, rows = the
     usable-and-valid sample indices); a run that merely touches the window but reads no usable sample takes no part.**
     Rows computed before are bit-identical (each segment's draws come from its own seed key);
     rows the extra segments refused are now computed. Measured before/after on two synthetic fixtures (185,310 pair
     rows, period and possession windows): every previously computed null and every other column bit-identical; 114
     SkillCorner cross-correlation rows moved from `segment_too_short` to `computed`, nothing else.
   - **Handover tolerance (owner ruling 2026-10-03).** A count excursion that returns to its prior count within
     `max_stoppage_s` no longer splits the team segment (GradientSports substitutions; above).
   - **The possession spectrum is sliced like every team signal (2026-10-03).** Spec 7.8.4 gives the possession
     series "the same treatment" (window ∩ segment slices); it was sliced only by where a possession was known, so it
     ran across long stoppages and count changes.
   - **Surrogate seed keys use the canonical game id (review minor 7, spec 7.9).** `str(game_id)` re-keyed every draw
     for a float-typed id (1.0 vs 1); canonical ids are unchanged for int and string ids, so no corpus value moves.
   - **Reported values:** the `n_runs_too_short` counter (4x -> 1x) and `resample_frames`' game entity (above); the
     orchestrator's merged report now counts the spectral and RSI rows too (`rows_by_source`; the two families never
     handed their counts over); and `CoordinationCoverageWarning` is spec 7.13's ONE call-level trigger -- the dropped
     share of WINDOWS in the call, warned by each public compute and once by the orchestrator, attributed to the
     caller -- not a pair-row share raised up to five times from inside the module.

   - **Round-2 review fixes (2026-10-04), each a corrected-output change on the noted rows:**
     - **A-06 spectral median frequency.** The periodogram's cumulative power is now linear between bin EDGES (a pure
       tone on bin k returns exactly k·Δf, spec 9.1); it read half a bin low before. Every `coord_median_freq_cpm`
       and the D1-derived `band_low/high_cpm` move.
     - **A-07 bridge-then-filter.** A detection gap is bridged at the native rate BEFORE the Butterworth (spec 7.4
       steps 2-3); it filtered the detected samples as if evenly spaced then bridged, distorting every bridged gap in
       every SkillCorner player series by up to metres.
     - **A-48 exact pre-warped Butterworth cutoff (coordination opt-in).** The signal-prep filter designs through
       `exact_prewarped_cutoff`, so the combined dual-pass −3 dB lands exactly at the cutoff (the linear-Winter
       default drifted with the bilinear warp: ~0.18% at 0.4 Hz, ~9% at 3 Hz). Every filtered coordination value
       moves (sub-% at the interim cutoff). Default path untouched (velocity/smoothing bit-identical); the repo-wide
       default flip is a filed owner-gated follow-up (TODO).
     - **A-08 one detection construct.** All eight families emit `insufficient_detection` on the MIN side raw-detected
       share (`coord_detected_share`, new on all seven tables); four families never tested it before and the other
       four tested a side-mean ~1.0 for dyads. Empty window → `too_short`; fully observed never flagged. D1 re-derives
       `min_observed_fraction` (new column; supersedes the 2026-09 "both scored fully observed" occlusion ruling).
     - **Degraded parents keep tokenised children (A-21 + pair-phase).** A degraded cluster-team window now emits its
       roster's player rows and a degraded pair row its `n_phases` phase rows, NaN metrics + the parent's token
       (ADR-042); both dropped them before.
     - **A-22 window `n_phases` governs.** Period/sliding windows (NA) emit no pair-phase rows; every window got 3
       before. A-24 flags `compactness_x` and the spectral path `goal_end_unresolved` when the end is unresolved.
     - **A-23 pair order A = attacking.** Cross-team dyads and the relative stretch index order (attacking, defending)
       on possession windows (spec 7.7); they used canonical order. `rsi_switch_times` (period-only) is unchanged.
     - **A-20 SampEn source columns.** `coord_*_sampen_source` split SampEn's token from the row's; an undefined
       SampEn no longer demotes a valid `rho_group`, and a constant spectral slice / empty coherence band is
       `degenerate_constant`/`too_short`, never `scored` with a NaN.
     - **A-25 SampEn within runs + φ_k unwrap.** Templates never span a gap (summed per-run counts); φ_k is unwrapped
       per run before its (linear) SampEn. Every `coord_*_sampen` on gapped/wrapped data moves.
     - **A-30 no-row windows.** A window that emitted no row is counted under `no_row`, not silently as scored.
     - **A-14 `max_detection_gap_s` noise floor.** D1 compares the bridged-position RMSE against the residual-analysis
       (Winter) intercept, not the residual at the fixed interim 0.4-Hz cutoff; the derived gap moves.
     - **A-52 welch_segment_s / RESIDUAL_GRID.** Shortest-meeting-resolution (spec amended to match the code);
       RESIDUAL_GRID now spans [0.1, 5.0] inclusive. D1 output only.
     - **A-17 IAAFT refused for cluster/team-sync.** No value moves (no driver sets IAAFT), but the option now raises
       for those families instead of silently time-shifting and reporting `computed`.
     - **A-45 phase-valid fraction drops the run-first sample.** `coord_rp_phase_valid_fraction_a/b` counted each
       run's first sample as non-advancing (it has no predecessor), biasing the fraction down by `n_runs / n`. The
       advance test is now one kernel (`phase_advance_indicator`): the first sample of every run is NaN and excluded
       from both numerator and denominator (`nanmean`). Every relative-phase row's two valid-fraction columns move up
       slightly; `phase_valid_fraction`'s scalar is unchanged.
     - **A-51 coupling-angle series mirrors vector coding's admission.** `compute_coordination_series(kind=
       "coupling_angle")` emitted angles for pairs vector coding never scores — dyads (no `vector_coding` method) and
       non-commensurate pairs. It now admits only VC-eligible, commensurate pairs; the ~1.1M spurious dyad rows are
       gone. Raw primitive, not a contracted table; relative-phase series unchanged.
     - **A-39 D1 pass-B estimators match the metric.** The decorrelation estimate is now computed PER stationary
       segment (`_acf_zero_over_segments`), never over the concatenation of non-adjacent runs; the spectral median skips
       a segment shorter than `min_spectral_samples` (the metric's own floor, at the interim band). So D1's derived
       `band_low_cpm`/`band_high_cpm`/`min_shift_s` move. The floor uses the INTERIM band (pass-B runs on default
       params, like every pass-B estimate) -- a methodology choice flagged for owner confirmation in round 3.
     - **B m1 multi-step handover absorption.** `_absorb_handovers` now absorbs any count excursion that leaves a level
       and FIRST returns to it within `max_stoppage_s` (a staggered double substitution 11→10→9→11), not only a mirror
       step 11→12→11. Segmentation moves on matches with such handovers (none on the 50-match subset; the full corpus
       may carry them); a non-returning change or one touching the period edge is still kept.
     - **B m3 cluster null keys on contributing runs.** `_cluster_surrogate` now refuses `segment_too_short` only for a
       player run that holds a usable `rho_group` sample the statistic reads (`_segments_holding`, rows = usable-and-valid
       indices), like the pair families — not for any run merely touching the window. A touching run that reads no usable
       sample moves `segment_too_short → computed`; a contributing run shorter than `2τ+1` still refuses. Shipped
       cluster-null rows move; the shift unit (player runs) is unchanged.
     - **A-19 coverage sample counters populated.** No shipping metric moves; the REPORT's own diagnostics change.
       `samples_unobserved`/`samples_stationary`/`samples_below_min_players` were declared, initialised and never
       incremented (a made-up 0). Now: `samples_unobserved` counts team samples voided because an on-pitch player has no
       bridged position (signal build); `samples_stationary` the vector-coding diffs omitted as stationary; and
       `samples_below_min_players` the cluster samples with the team present but fewer than `min_players` valid phasors.
     - **A-55 no-flip verdict honesty.** No shipping metric moves; the GATE's own report changes. `n_values_compared`
       now counts only cells finite in at least one leg (a NaN–NaN cell of the melted union is not a comparison), with
       `n_cells_total` beside it; source/token columns still compare every row. The verdict also records `run_tree_hash`
       (`_provenance.git_tree_hash`: a throwaway `GIT_INDEX_FILE` of `read-tree HEAD` + `add -A` + `write-tree`), tying
       the numbers to the exact working-tree CONTENT, not only to a commit + a dirty flag.

     - **A-09 per-construct reliability + occlusion (round-2 cycle).** The D3 `metrics.json` reliability section is
       rebuilt at the **per-construct** grain (a metric column × its C.1 keys; ~2000 cells) — replacing the author's
       one-representative-metric-per-family block. Each cell carries a binding reliability (linear ICC(1), or
       **rotation-invariant circular** reliability for the three circular-mean constructs `coord_rp_mean_deg` /
       `coord_vc_mean_angle_deg` / `coord_phi_mean_deg`), a group-bootstrap 95% CI, a pre-registered **power verdict**
       (`RELIABILITY_MIN_N_GROUPS` / `RELIABILITY_MAX_CI_HALFWIDTH` / `CIRCULAR_RELIABILITY_MIN_RBAR`; an underpowered
       cell is terminal "unmeasurable", never pooled), a split-half **Spearman–Brown** block, deciles and cross-provider
       poolability, plus the two §C.4 honesty lines. D1's `min_observed_fraction` is re-derived **per construct** and
       consumed per family as the fail-closed **MAX** over the family's constructs' qualifying shares (owner ruling
       below); `derivation.json` gains `occlusion.per_construct` (curves + n/CI per bin + estimable verdict + binding
       construct + the over-restriction diagnostic). Every reliability/poolability/decile/occlusion number in
       `metrics.json`/`derivation.json` and the committed generated module moves on regen; the no-flip SIGNAL gate is
       unaffected (it compares signal numerics).
     - **A-34 / A-35 / m7 D2 objective.** The OAT objective keeps per-fold vectors by INDEX with NaN for an unscoreable
       fold (A-34), scores a circular-mean construct with circular reliability and never a linear ICC (A-35, the SAME
       per-construct estimator D3 reports — the team-discrimination vs within-match-consistency grain difference is a
       ruled reconciliation stated in `metrics.json`), and keys CV folds on the provider-qualified `join_key` (m7).
       `OBJECTIVE_VERSION` 1→2, so no stale store resumes; D2's selections may move on re-run.
     - **A-36 SMA/RMA label.** `scripts/_reliability.type_ii_slope` is documented as the standardised/reduced
       major-axis slope (it was mislabelled "orthogonal"); the arithmetic and pinned value are unchanged.
     - **A-53 paired tests.** The sign test excludes ties from its trial count and TOST calls an all-identical
       difference set within ±margin equivalent; H1/H5 p-values may move where ties or zero-variance halves occur.

  **Occlusion-grain decision (owner, 2026-10-04).** `min_observed_fraction` is consumed per family (its gate is
  inherently per-family: `params.min_observed_fraction[family]`, one float), as the fail-closed MAX over the family's
  constructs' qualifying shares — correctness identical to a per-construct gate (MAX ≥ every construct's own share),
  only retention differs. The public `CoordinationParams.min_observed_fraction` stays `Mapping[family → float]` (no
  seam change, `_compute.py` untouched). The full per-construct analysis and an **over-restriction diagnostic** (how
  much more strictly the MAX gates than each construct's own share; any family driven to 1.0 by a single construct) are
  recorded in `derivation.json` as the owner's decision input. **Follow-up (owner-gated, costed here):** true
  per-construct *consumption* — a private per-construct threshold table behind the gate, leaving the public dataclass a
  family-MAX summary — is taken ONLY if the over-restriction diagnostic shows the retention loss is material; it widens
  no public API and changes no correctness (MAX is already fail-closed). Full per-construct consumption by widening the
  public dataclass to ~thousands of entries was ruled OUT of this cycle.

  Each is recorded here, and the corpus no-flip gate runs on the corrected signals (both of its legs share them).

**DGX ordering (owner, 2026-09-29).** The authoritative full-corpus no-flip runs on the clean commit-1 tree (plan
Task 27), after the diff-approval stop: local subset trial -> `/final-review` -> ADR numbering -> the owner approves
commit 1 on the feature branch -> the DGX gate against that exact commit -> nothing merges to `main` until it passes
(and until the owner's other cycle + DAS land). A local no-flip trial on a corpus subset de-risks it beforehand.

## ADR amendments

- **ADR-048 (feature glossary / attribution), R5:** the `Unit` vocabulary gains `"cycles/min"` (spectral median
  frequency). Every published-methodology coordination column carries a `NOTICE` attribution paragraph; the C4 model
  gains a `coordination` container pinned to the code.
- **ADR-098 (metric-family output contracts):** the completeness test is re-keyed **per exported constant** so the
  seven coordination families (`coordination_pair`, `_pair_phase`, `_spectral`, `_cluster_team`, `_cluster_player`,
  `_team_sync`, `_rsi`) each register or the gate fails.

## Related

- **Specs:** `docs/superpowers/specs/2026-09-26-tf58-team-coordination-design.md` (owner-approved; reviews r1–r3).
- **Plans:** `docs/superpowers/plans/2026-09-26-tf58-team-coordination.md` (approved r4).
- **ADRs:** depends on ADR-106, ADR-107, ADR-108, ADR-109; amends ADR-048, ADR-098; selection criterion ADR-060.
- **Reviews:** `D:\Development\_reviews\2026-09-28-tf58-team-coordination-{d1-impl,d1-impl-r2,d2-impl,d2-impl-r2,d3-impl}.md`.
- **External references:** Bourbousson 2010; Moura 2012/2013/2016; Duarte 2013; Folgado 2014; Richardson 2012;
  Shewchuk 1997 (orientation predicate); Winter 2009 (residual analysis).
