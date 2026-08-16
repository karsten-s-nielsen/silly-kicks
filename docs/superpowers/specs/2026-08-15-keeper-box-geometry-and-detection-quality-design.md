# Keeper-box geometry & detection-quality cycle — design

**Date:** 2026-08-15
**Status:** Draft — reviewed twice (part-deux session); round-1 concerns fixed, round-2 recommendations
adopted, all owner decisions resolved (standalone clamp confirmed). Pending: final owner sign-off to move to
writing-plans. Not committed.
**Spine chosen by owner:** full keeper-geometry pipeline (option B), one PR, minimal commits, pining-sourced.
**Numbering:** version / PR / ADR numbers are assigned at commit-prep, after merging `origin/main` — deliberately NOT fixed here.

---

## 1. Context & motivation

Three pieces of work were left standing after the 4.81.0 ghost-refit / TF-24 Stage-1 cycle. They are
independent in their logic but share one substrate — the pining-sourced 179-match tracking corpus and
the confirmed Stage-1 carrier params — and one of them (the `gr_x` decision) is gated on a measurement
that overlaps almost exactly with another (SkillCorner off-pitch detection quality). Running them as one
cycle on one materialized corpus is cheaper and more coherent than three separate passes.

The three pieces:

1. **SkillCorner keeper-origin real-data validation + standing CI rate-gates.** The S1–S4 resolver
   changes shipped in 4.37.0 (ADR-024); what is outstanding is *confirming them on real pining data* and
   *landing the two standing rate-gates* the TODO records. Its off-pitch characterization is the input to
   piece 2.
2. **The `gr_x` behind-the-line decision (ADR-050 §6).** `in_penalty_area_goal_relative*` has no lower
   bound on `gr_x`, so points behind the goal line (`gr_x < 0`) count as in-box. 4.81.0 measured the
   population (233,359 rows inside the y-band, 0.058% of 401.9M) and *deliberately deferred* the decision
   to keep the constant re-fit attributable. This cycle takes the decision on a number.
3. **TF-24 Stage-2 tracking-defaults calibration refresh.** Stage 1 closed in 4.81.0 (ADR-060). Stage 2's
   `k3` / `pre_seconds` / `min_displacement_m` still ship as engineering defaults TF-24 never set; re-sweep
   the augmented-VAEP Brier objective on the corrected-geometry corpus, holding the confirmed carrier params.

Plus a recurring ADR-code reconciliation sweep at ship.

### Key structural finding (verified, not assumed)

**TF-24 Stage-2 is fully decoupled from the `gr_x` clamp.** The Stage-2 augmented-VAEP feature matrix
(`silly_kicks/calibration/_features.py::ALL_FEATURES`, 7 SPADL + 48 tracking columns) contains **none** of
`_ghost_gk`'s or `_xcross_attempt`'s features, and those two modules are the *only* consumers of the
signed-`gr_x` predicate (`in_penalty_area_goal_relative*`; `in_penalty_area_absolute` abs-folds first, so
`defensive_credit` is unaffected). Therefore the clamp cannot move a single Stage-2 feature, and the three
pieces share only the corpus — there is **no strict pipeline ordering**.

---

## 2. Goals / non-goals

**Goals**
- Confirm the shipped SkillCorner keeper-origin resolver on real pining data; land two standing CI rate-gates.
- Decide the `gr_x` behind-the-line predicate on a pre-registered, materiality-driven rule (basis A), and
  execute whichever branch the data selects.
- Refresh the TF-24 Stage-2 recommendation on corrected geometry; surface (not adopt) any default change.
- Keep the whole cycle to one PR and the minimum number of provenance commits.

**Non-goals**
- Re-implementing S1–S4 (already shipped 4.37.0). This cycle validates and gates them.
- Changing any library default constant for TF-24 (ADR-009: the harness recommends; adoption is a separate PR).
- Wiring `visible_area` coverage into count features (ADR-055 / ADR-009 line; needs a consumer asking).
- Any lakehouse mart / `access_tier` plumbing (ADR-024 boundary).
- The xShot (TF-16) model — it does **not** consume the `gr_x` predicate, so it is out of scope and not re-fit.

---

## 3. Architecture: three pieces on one substrate

```
pining-for-the-data  ──►  materialize tc3 cache (within-run)  ──►  ┌─ Phase A: SkillCorner validation driver ─► research artifact + 2 CI rate-gates + off-pitch characterization
   (all providers)         [scripts/_loader_pining_to_cache.py]    ├─ Phase B: gr_x measurement (extends measure_box_constant_delta.py) ─► pre-registered 3-outcome decision
                                                                    └─ Phase C: TF-24 Stage-2 sweep (calibrate_tracking_defaults.py --stage 2) ─► recommendation manifest
```

- **Data source.** Every driver sources via `pining-for-the-data`; the downloaded provider folders are not
  an input. The tc3 frame cache is a *within-run* materialization from pining (allowed; the rule is about
  the source of truth, not about avoiding cache). All new/extended drivers adopt the `scripts/_driver.py`
  `for_each` shard seam (ADR-052) and the `scripts/_provenance.py` clean-tree + `run_commit` discipline.
- **Compute.** Owner token + full DGX access are held by the implementer, who runs every probe and corpus
  pass (local for fast iteration, DGX for real slices and the full corpus). The owner approves each commit
  and publishes the release.

---

## 4. Phase A — SkillCorner keeper-origin validation + CI rate-gates

**Prior art:** S1 (coordinate transform), S2 (per-player `is_visible` on the frame model), S3 (tiered
keeper-origin resolution `{tracking_gk, goalkick_prior, unresolved}`), S4 (loud validation of a native
goal-kick origin implausibly far from goal) **shipped in 4.37.0 / PR-S104 / ADR-024**. This phase does not
re-implement them.

**4.1 Real-data validation (reported, not gated).** A new driver
(`scripts/validate_skillcorner_keeper_origin.py`, adopting `_driver.py`) runs the shipped resolver over the
real pining SkillCorner corpus and produces a research artifact under
`docs/research/skillcorner_keeper_origin/` confirming the ADR-024 acceptance criteria:
- goal-kick origins ≈100% own-box; the scatter SD collapses from the ~23.2 m broadcast baseline;
- open-play pass origins localize; the `unresolved` subset is produced, countable, and never imputed;
- per-tier provenance (`xt_gk_origin_source`) populates as designed.

**4.2 The two standing rate-gates — split by population (C2).** A corpus-derived threshold applied to a
few-match slim fixture is a population mismatch (vacuous if the margin is loose, flaky if it is tight), so
each gate exists at two altitudes:
- **(a) A generous STRUCTURAL gate on the committed slim fixture — every leg, not `@e2e`.** Asserts the
  rate is *computed, finite, and under a deliberately loose ceiling* — plus a **mandatory both-sides
  mutation** (a fixture perturbation that pushes the rate over the loose ceiling and is asserted to fail).
  This proves the gate is wired and non-vacuous; it does NOT assert the corpus baseline.
- **(b) The TIGHT corpus-baseline gate as an owner-run `@e2e` / data-contract** on the full pining corpus,
  threshold = *measured baseline + margin*, recorded with `run_commit`. This is where the real
  off-pitch-rate and out-of-region-goal-kick-rate assertions live, on the population they were calibrated on.
- The two quantities are the S1 gross-off-pitch rate (ADR-024 S1: a gross off-pitch under a correct
  transform must fail loud, never silently clamp) and the out-of-region native goal-kick rate (ADR-024 S4).
  "Runs on every leg" (a) and "asserts the corpus baseline" (b) are deliberately not the same gate.

**4.3 Two smaller items the TODO row bundles here:**
- **Validate-then-maybe** the open-play own-half misdetection bound: add a generous own-half bound *only if*
  pass origins still land in the attacking half; beyond it → `unresolved`, never clamped.
- **Measure-before-optimize** the `_tracking_gk_xy_detected` ±window loop before touching it.

**4.4 The off-pitch characterization (Phase B's cross-check).** The same driver emits, per behind-line
row, the distance-to-goal, lateral offset, `is_visible`, and whether `_loader_pining` flagged it as
off-pitch beyond the S1 tolerance. This is the `offpitch_fraction` input the `gr_x` decision consumes — one
measurement serves both pieces.

**Regression guard:** GS / idsse / metrica keeper-origin resolution + values must be byte-identical
(ADR-024 acceptance; this is SkillCorner-only).

---

## 5. Phase B — the `gr_x` behind-the-line decision (basis A)

**5.1 What is being decided.** `in_penalty_area_goal_relative_array(gr_x, y)` is
`(gr_x <= penalty_area_depth) & (|y − 34| <= penalty_area_half_width)` with no lower bound on `gr_x`.
Points behind the defended goal line (`gr_x < 0`) satisfy `<= depth` and count as in-box. The signed `gr_x`
reaches exactly two trained features: ghost's `attackers_in_box` and xCross feature #6 (box off/def ratio).

**5.2 The measurement — R_M identical by construction (C4).** The measurement must run on each trainer's
*exact* training population, or the decision is made on the wrong rows (a subtly different row set still
produces a plausible number — a silent error). Verified: both extractions are already callable seams —
`prepare_ghost_gk_training_data` (public, `silly_kicks.tracking`) and `prepare_xcross_training_data`
(`silly_kicks.tracking._xcross_attempt`) — so no refactor is needed. The extended
`scripts/measure_box_constant_delta.py` therefore, for each model M ∈ {ghost, xCross}:
1. calls M's extraction seam **twice** on the pining corpus — once under the shipped predicate, once under a
   **measurement-only scoped clamp** of `in_penalty_area_goal_relative_array` (a context manager applying
   `gr_x >= 0` for the duration of the second call; this ships NO clamp — the real clamp is commit-2, §5.4);
2. diffs the box-feature column (`attackers_in_box` for ghost, feature #6 for xCross) between the two runs:
   `changed_fraction_M` = |{ r ∈ R_M : column differs }| / |R_M|, plus the magnitude distribution;
3. emits `offpitch_fraction_M` = among changed rows, the fraction flagged off-pitch by `_loader_pining` (Phase A);
4. emits the **training-row base rate** of behind-line-in-band (`gr_x < 0 ∧ in y-band`) on R_M (C3 — see §5.3);
5. emits the **distance-to-goal histogram** of the changed rows — the third lens the original TODO asked for:
   flip *fraction* is a proxy for whether the model's output moves *where the feature carries signal*, so
   concentration near the attacked goal (real shot context) vs a thin midfield spread (noise) is what
   distinguishes a material geometry error from sensor jitter (§5.3 D-geom).

Because both calls go through the identical seam on the identical corpus, R_M and the feature computation are
identical by construction and the only difference is the predicate. A fixture test asserts the measured
population equals the trainer's row set on a known slice (guards against the seam drifting apart later).

Two implementation pins for the scoped clamp (round-2 N1/N2): (a) patch the **module attribute**
`_geometry.in_penalty_area_goal_relative_array` — both consumers call it via attribute access, so a patched
attribute is seen by both; do not patch a re-imported name. (b) The measurement clamp applies the **same
bound value** as the real clamp would (`penalty_area_min_gr_x`, 0.0 today), enforced by a test that the two
produce identical box features, so a future move of the constant off 0.0 cannot make the measurement predict
a delta the real clamp does not produce.

**5.3 The decision — TWO orthogonal decisions, not one (round-2 reframe).**
The `gr_x < 0` symptom has two distinct defects behind it, and the round-1 framing wrongly let one veto the
other:

- **A geometry-correctness bug.** `in_penalty_area_goal_relative*` counts a point *behind the goal line*
  (`gr_x < 0`) as in-box. The penalty area is Law-bounded by the goal line, so this is wrong **as geometry**
  — and wrong for a *real* behind-line position exactly as much as for a sensor artifact. Fix = the clamp (§5.4).
- **A data-quality problem.** Off-pitch sensor detections exist and corrupt *every* raw-xy feature (pitch
  control, pressure, defensive line, …), not just the box predicate. Fix = the ingestion seam.

They are gated on their own evidence, independently.

**(D-geom) Clamp the predicate iff it is MATERIAL and the geometry is demonstrably wrong for REAL positions.**
- *material:* `changed_fraction_M ≥ τ` for at least one M (τ = 3.62e-5, the constant-unification floor — its
  job is to catch the genuinely-immaterial case so no published artifact is touched for noise);
- *real-positions-exist:* among the changed rows a non-negligible population is *real* near-line positions
  (small |gr_x|, not `_loader_pining`-flagged off-pitch), evidenced by the **distance-to-goal histogram**
  (§5.2 step 5): concentration near the attacked goal is where `attackers_in_box` carries signal; a thin
  spread is noise.
- If both hold, the geometry is wrong for real players in real situations → clamp. **A high off-pitch
  fraction does NOT veto this** — the clamp is a strict correctness improvement regardless of any row's
  origin; it simply does not *also* clean the data (that is D-data).
- Pure no-clamp is correct only if the change is immaterial OR the real behind-line population is negligible
  (behind-line points are essentially all garbage). Checkable from the same measurement, and it may differ
  per model — plausibly closer to all-garbage for ghost (attackers rarely stand behind the opponent goal
  line) than for xCross (its off/def ratio includes keepers/defenders, who *do* occupy the goal-line region
  during shots). The clamp is one shared predicate: if warranted for either model, clamp both.

**(D-data) File the ingestion-seam TODO iff off-pitch contamination is material — independently of D-geom.**
High off-pitch is its own signal; it neither triggers nor blocks the clamp.

**Execution of a warranted clamp — standalone now vs ride the next re-fit.** The clamp's entire cost is the
re-fit (a second ghost re-fit in two releases + republish). The churn-minimizing path is to bundle it into
the *next scheduled* ghost/xCross re-fit at zero marginal cost — but **only if one is actually coming**;
deferring into an unscheduled future ships the known-wrong predicate indefinitely, which is not acceptable.
No ghost/xCross re-fit is currently scheduled, so a warranted clamp is executed **standalone in this cycle**
(commit-2). If the owner knows of an imminent re-fit, the clamp rides it and this cycle ships the decision +
ADR only. This is the one residual owner sub-decision (§12).

**On τ and φ (round-2 Decision 1).** τ = 3.62e-5 is kept as a *floor* (catch the immaterial case); §5.2 step
4 reports whether it binds on the box-enriched training population — it likely will not, which is fine: a
cheap "don't bother below this" floor is still worth having, and the substantive test was never a bare
fraction. **φ = 0.5 is retired as a clamp gate** — it was internally inconsistent (it would clamp even at
49% garbage while the rationale said *any* material off-pitch means the box is the wrong seam) and D-geom no
longer uses it. φ survives, if at all, only as the D-data ingestion-TODO trigger, where a *low* threshold
(≈0.2–0.3, or simply "material off-pitch present") is the honest test, not a majority vote. All thresholds
are pre-registered before the DGX run and applied mechanically to the provenance-stamped artifact.

**5.4 The clamp, if D-geom is warranted — a DECLARED CONSTANT, landing atomically in commit-2.**
The entire clamp mechanism — the new `spadlconfig` constant, the predicate reading it, the `cache_token()`
wiring, the feature-contract declaration, the re-fit, and the re-stamp — lands together in a **single
commit-2**, and *only* if D-geom is warranted and executed standalone (§5.3). **Commit-1 ships no clamp at
all** (the measurement's
counterfactual is a scoped, measurement-only patch, §5.2), so the shipped models and their contracts are
untouched until the decision is taken.

This is load-bearing and corrects the ADR-050 README's own framing. A bare `gr_x >= 0` clamp is invisible to
*two* mechanisms:
- the ADR-050 feature contract (declares no new constant → `_feature_contract_block()` byte-identical → no CI backstop; measured);
- `train_ghost_gk.py::cache_token()`, which derives from the geometry constants only → unchanged under a bare
  clamp → the re-fit would **silently reuse the un-clamped features** while stamping a "clamped" contract.

Implementing the lower bound as a named constant in `spadlconfig` that the predicate reads, the
`cache_token()` derives from, and the feature contract declares, converts the blind predicate-shape change
into a *visible constant change* — it rides the existing cache-token and feature-contract mechanisms instead
of defeating them. The `in_penalty_area_goal_relative_array` body gains `& (gr_x >= <const>)`. The name must
encode the axis and inclusivity — proposal `penalty_area_min_gr_x = 0.0`, documented that it is a
**non-strict** lower bound so a point exactly on the goal line (`gr_x == 0`) counts as in-box, matching the
Law and the existing non-strict `<= depth` upper bound. **Verified:** the scalar
`in_penalty_area_goal_relative` delegates to the array body and `in_penalty_area_absolute` delegates to the
scalar (`_geometry.py:135, :201`), so the bound is added in exactly one place; `in_penalty_area_absolute` is
unaffected in value (abs-folded `gr_x` is never negative).

**5.5 The re-fit (D-geom warranted, standalone).** Re-run `train_ghost_gk.py` and `train_xcross_attempt.py` on the corrected
predicate (pining-sourced), preserving each model's existing gates (ghost chirality + feature contract;
xCross paired data-effect + fail-closed acceptance + its own contract), re-stamp the feature contracts on
x86 (`scripts/stamp_feature_contracts.py`), and republish both to the Hub. The ghost re-fit inherits the
4.74.0 platform `atol` baseline (`docs/research/pr5_platform_atol/`) — re-measure only if the ghost
extractor's own probe changes. Note the box constant re-fit already shipped in 4.81.0; this is a *second*
re-fit of ghost within two releases, which is exactly why outcome 1 must be *earned* by the measurement,
not assumed.

---

## 6. Phase C — TF-24 Stage-2 refresh

**6.1 The sweep.** Run `scripts/calibrate_tracking_defaults.py --stage 2 --source pining --providers
skillcorner idsse gradientsports`, holding the confirmed Stage-1 carrier params, minimizing the
augmented-VAEP held-out Brier over `k3` / `pre_seconds` / `min_displacement_m` (`stage2_config`).

**The carrier handoff is a resolved contract, not a plan detail (C6).** `--carrier-best` takes the
provenanced 4.81.0 `carrier_selected.json` (its help text already says so). `_load_carrier_selection`
(`calibrate_tracking_defaults.py:253`) **already enforces ADR-060**: it sources `tolerance_m` from
`DEFAULT_CARRIER_PARAMS` (not the file), and **refuses** a selection with no `run_commit` ("an unprovenanced
upstream is treated as dirty", `:267`) or with `run_tree_dirty is not False` (`:269`). Phase C confirms a
red-first refusal test covers both branches (dirty → raise; unprovenanced → raise) and adds one if absent. Frozen exogenous xT is the
disjoint held-out artifact the harness already builds. Produces the report + data/version manifest with
provenance.

**6.2 The decision.** Per ADR-009 the harness *recommends* and never changes a library constant. This phase
surfaces the recommendation and its Brier delta vs the incumbent engineering defaults in the manifest; any
adoption of a recommendation as a default is an explicit separate PR, out of scope here. This also subsumes
the optional `δ` / `MIN_EFFECT_SIZE` principled-derivation completeness item (δ derives from the same
Stage-2 Brier sensitivity; the δ-invariance check already discharged the risk across `[0, 0.1]`).

**6.3 Independence.** Decoupled from Phase B (§1 finding), so Phase C runs against the same commit as Phase A
regardless of the `gr_x` outcome.

---

## 7. Phase D — ADR-code reconciliation sweep

The recurring once-per-minor consistency check: verify documented ADRs still match the codebase (stated
constraints hold, superseded decisions updated). Folded into the ship commit; no separate artifact.

---

## 8. Commit & verification protocol (contract for this cycle)

This section is a written contract, not a preamble.

1. **Every commit is a finished, locally-CI-green, substantial unit — verification SHOWN, not claimed.**
   Before requesting approval for any commit, run and paste the real exit-coded output of:
   `python -m pytest tests/ -m "not e2e" --benchmark-skip`, `python -m ruff check silly_kicks/ tests/ scripts/`,
   `python -m ruff format --check silly_kicks/ tests/ scripts/`, and bare `python -m pyright`. No commit
   rests on a claim; piped runs that mask exit codes are not evidence.
2. **Two pre-commit gates for every data-touching driver, both pasted:** a committed-fixture test (proves the
   code path) *and* a real-data pining probe (proves it survives real data and the deployment environment —
   a match or two, on DGX for the SkillCorner and full-corpus drivers). The probe must demonstrably pull via
   pining, not a downloaded folder.
3. **Commit count = the provenance floor and nothing above it.** No WIP, no fix-up commits. A defect is fixed
   before the commit, not after. Each commit is approved against a real `git diff --stat` + message at the
   moment of committing.
4. **New gates land red-first** — observed failing before the fix exists.
5. **Non-squash merge** so stamped `run_commit` SHAs survive on `main` (commit-policy provenance rule).
6. **The spec and plan are committed WITH the relevant code commit, never standalone.**

---

## 9. Sequencing & commit map

Three DGX passes (A investigation, B measurement, C Stage-2) are mutually independent and run against the
same first commit.

- **commit-1** — Phase-A validation driver + the two CI rate-gates + Phase-B measurement extension of
  `measure_box_constant_delta.py` + Phase-C run readiness. → run DGX passes A, B-measurement, and C against
  this SHA; artifacts stamp it.
- **commit-2** — the `gr_x` decision (§5.3):
  - D-geom **not** warranted → doc-only (research note + the D-data ingestion TODO if off-pitch is material +
    ADR); folds toward the release commit.
  - D-geom warranted, standalone → the declared-constant clamp + ghost & xCross re-fit + re-stamped contracts
    + republished weights; its own provenance SHA. (If instead riding a future re-fit, this collapses to the
    doc-only case plus a queued clamp decision recorded in the ADR.)
- **commit-3 / release** — land Phase-A/C artifacts, the ADR amendments, CHANGELOG, TODO grooming, version
  bump, and the ADR-code sweep.

**Commit count: 2 if the clamp does not fire, 3 if it does. One PR.**

**Why one PR / minimal commits, not one-concern-per-commit (C1).** The §3 decoupling is a *compute*
property (materialize once, run three passes) and an argument that the pieces *can* run in any order — it is
not an argument that they must ship separately. The owner's standing directive is minimal PRs and minimal
commits, and the recent-history failure this cycle is reacting to was WIP/fix-up commit churn. `commit-1`
bundles three concerns, but each is independently complete and CI-green before the commit (§8), and the
commit boundaries fall on the DGX-provenance handoffs, not on WIP. Splitting `commit-1` three ways to raise
reviewability would reintroduce exactly the commit count the owner objects to; reviewability is instead
served by the PR's phase structure and this spec/plan. **Decided (owner, 2026-08-15): one PR + minimal
commits, `commit-1` not split.**

---

## 10. Testing & CI

- **Phase A:** the two STRUCTURAL rate-gates on committed slim SkillCorner fixtures (all legs, not `@e2e`) —
  each asserts computed/finite/under-a-loose-ceiling **plus a mandatory both-sides mutation** (a perturbation
  that breaches the ceiling, asserted to fail; §13); the TIGHT corpus-baseline assertions are the owner-run
  `@e2e` data-contract (§4.2b). A per-tier golden already exists from 4.37.0 and stays green;
  GS/idsse/metrica regression byte-identical.
- **Phase B measurement:** the existing `test_measure_box_constant_delta.py` conservation guard extends to
  the new training-feature-delta columns (per-cause attribution, not aggregate); a fixture test asserts the
  measured population equals the trainer's row set on a known slice (§5.2, C4).
- **Phase B clamp (outcome 1 only):** a **red-first** test that `cache_token()`'s value differs before/after
  the constant is wired into its format string, AND that the feature-contract declared-constant block
  changes — observed failing first, so §5.4's anti-silent-reuse guarantee is exercised, not just asserted
  (C5); the declared constant enters `test_geometry_constant_enumeration.py` (every geometry constant
  declared-or-exempt); `test_geometry_box_predicate_parity.py` re-pinned to the new bound (red-first: observe
  it fail on the old body); ghost chirality + feature-contract load guards; xCross paired-effect +
  fail-closed acceptance; the box-predicate migration identity test updated.
- **Phase C:** cache-equivalence (`assert_cache_equivalence`, 1e-9); a red-first test that Stage 2 refuses a
  carrier selection with `run_tree_dirty=True` and one with no `run_commit` (C6); no library-default change,
  so no golden moves.
- **CI faithfulness:** the full non-e2e suite on all legs plus the primary-leg-only slow/e2e set, per the
  ADR-023 partition; lint at CI scope; bare pyright. Numbers claimed in any artifact carry `run_commit` +
  `run_tree_dirty`.

---

## 11. ADRs, provenance, merge

- **ADR form:** a **new ADR** for the `gr_x` decision (records the basis-A rule, the measured outcome, and
  the declared-constant/cache-token mechanism), plus an **ADR-024 amendment** for the two rate-gates. The
  `gr_x` decision has a pre-registered rule and a measured basis — it is ADR-worthy on its own, and giving
  it a dedicated ADR avoids the "registered facts go stale together" pattern that an ADR-050 §6 amendment
  would risk. TF-24 Stage-2 is governed by ADR-009/ADR-060 and needs no new ADR unless a default is adopted
  (out of scope).
- **Provenance:** every artifact driver refuses a dirty tree and stamps `run_commit` + `run_tree_dirty`
  (`scripts/_provenance.py`); artifacts derived from another artifact carry provenance on both.
- **Merge:** `--merge`, never squash, because Phase-A/B/C artifacts (and, in outcome 1, the bundled weights'
  `training_commit`) stamp branch SHAs.

---

## 12. Risks & open questions

- **`gr_x` has no automatic CI backstop today.** §5.4's declared-constant implementation is precisely what
  removes that risk in outcome 1 (the change becomes visible to the cache token and feature contract). In
  outcomes 2/3 nothing changes, so the risk is not incurred.
- **The off-pitch-vs-real confound** is exactly what Phase A's `offpitch_fraction` resolves; the pre-registered
  φ = 0.5 threshold is the decision boundary and is open to adjustment before the run.
- **Second ghost re-fit in two releases** (outcome 1) — justified only if the measurement earns it; the
  materiality bar τ exists to prevent a no-op re-fit that churns published artifacts.
- **Platform-provenance gap** (aarch64-DGX vs AMD64) remains an open item; the ghost re-fit inherits the
  4.74.0 `atol = 1e-6` baseline and does not owe a new platform probe unless the ghost extractor's probe changes.
- **All owner decisions resolved — no open questions remain.**
- **Round-2 recommendations adopted:**
  1. **τ / φ** — τ = 3.62e-5 kept as a *floor*; φ = 0.5 **retired** as a clamp gate (survives only as the
     D-data ingestion trigger); the distance-to-goal concentration histogram added as the third lens
     (§5.2 step 5, §5.3).
  3. **Outcome 2 reframed** — geometry and data-quality are two orthogonal decisions; the clamp is gated on
     materiality + real-positions-exist, **not** on the off-pitch fraction; the ingestion fix is filed
     independently (§5.3 D-geom / D-data).
- **Owner decisions taken (2026-08-15):**
  2. **One PR + minimal commits** (C1) — confirmed; `commit-1` not split (§9).
  3-exec. **A warranted D-geom clamp is executed STANDALONE in this cycle**, on the same feature branch —
     confirmed; no re-fit is scheduled to ride (§5.3).
  4. **§4.3 scope** — the own-half bound and the ±window item **stay in this cycle**; single cycle, session
     may be restarted mid-cycle if needed.
- **Resolved from round-1 review:** ADR form (new ADR + ADR-024 amendment, §11); rate-gate population split
  (§4.2); R_M identical-by-construction via the trainer seams (§5.2); cache_token red-first (§10); the
  carrier refusal contract (§6.1, already coded); scoped-clamp pins N1/N2 (§5.2).

---

## 13. Discipline notes (non-vacuity)

- Phase A's structural rate-gates carry a **mandatory** both-sides mutation (not "where feasible"): a
  perturbation that should breach the loose ceiling is asserted to fail, because this repo's silent-null
  defects were caught specifically by the failing-side assertion. The tight corpus-baseline lives in the
  `@e2e` data-contract, pinned from a *measured* baseline + margin (§4.2).
- Phase B's decision is pre-registered before the run; the numbers are reported into a provenance-stamped
  artifact and the rule is applied mechanically to them.
- The `gr_x` measurement asserts the training-feature delta per model, not an aggregate, and the clamp's
  parity/enumeration guards land red-first so they are observed to guard something.
