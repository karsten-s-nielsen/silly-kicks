# TF-58 round-2 fix cycle — plan for all remaining work

**Date** 2026-10-04
**Branch** `feat/tf58-team-coordination` (off `main` @ `b62c1f2`; nothing committed).
**Status of this plan** DRAFT for `/review-plan` by the two independent sessions before the remaining work is implemented.

## 0. Purpose and bar

This plan governs the **remaining** work of the TF-58 round-2 review-fix cycle: closing the still-open CONFIRMED
findings of the two round-2 reviews, implementing the large per-construct reliability block (A-09), and routing the
whole cycle to the two independent sessions before commit 1 goes to the DGX.

The bar (source of truth, in precedence order):

- **Spec** `docs/superpowers/specs/2026-09-26-tf58-team-coordination-design.md` (amended through 2026-10-03). The spec
  wins over the plan.
- **Plan** `docs/superpowers/plans/2026-09-26-tf58-team-coordination.md` (the original; Task 27 DGX runbook, Task 28
  file list).
- **ADR** `docs/superpowers/adrs/ADR-111-team-coordination.md` — design decisions + the running
  **corrected-output list** (every behaviour change this cycle is appended there; see §F).
- **Round-2 reviews** `D:\Development\_reviews\2026-10-03-tf58-team-coordination-a-impl-r2.md` (Reviewer A) and
  `…-b-impl-r2.md` (Reviewer B). Each finding's "Resolved when" is its mini-spec.
- **Owner rulings** — four AskUserQuestion batches (2026-10-04); batches 1–3 are transcribed in-bar in **Part G** and
  batch 4 inline in Parts B/D/C.5. These settle the scope questions the reviews raised, and are in-repo (not only in the
  external scratch ledger).

Working aid only (NOT part of the bar, NOT the authoritative record — review R-1): the per-finding triage ledger
`scratchpad/merge/round2_triage.md` (session-scratch, target-external). It holds every finding, its ruling, and its
status; this plan points to it rather than re-transcribing the already-closed items.

**Standing constraints (owner, permanent):** nothing deferred/dropped/simplified below the agreed bar without explicit
approval; every fix is test-first (RED→GREEN); no commit without an explicit per-commit human-approval gate; one
feature branch, no worktrees; the DGX is owner-coordinated and currently in use (untouched).

## Part A — already implemented this cycle (pending round-3 review)

The CONFIRMED findings closed so far are tracked per-finding in `scratchpad/merge/round2_triage.md` (status `[x]` with
a one-paragraph record each: fix, tests, corrected-output). Reviewers should read that ledger + the working-tree diff
for these. This session's additions (each test-first, lint+pyright clean):

| Finding | Source | One-line fix | Corrected-output? |
|---|---|---|---|
| A-55 | A nit | no-flip `n_values_compared` excludes NaN–NaN cells; verdict records `run_tree_hash` | gate report only |
| A-51 | A nit | `compute_coordination_series(coupling_angle)` admits only VC-eligible (commensurate, non-dyad) pairs | raw-series only |
| A-45 | A nit | `phase_advance_indicator`: run-first sample NaN (not non-advancing); `coord_rp_phase_valid_fraction` ↑ | YES |
| A-39 | A major | D1 pass-B ACF per-segment (never spliced); spectral median honours `min_spectral_samples` floor | YES (D1 band/min_shift) |
| A-33 | A minor | C27 id + `input_contract` fold `_objective_fingerprint` (OBJECTIVE_VERSION + `_FAMILY_COLUMNS`) | store-id only |
| A-18 | A minor | family-level M-13 wiring: merge-monkeypatch per pair family + RSI-switch unit + possession-spectrum unit | test-only |
| A-42 | A minor | hand-counted SampEn/Cross-SampEn oracles + possession square-wave analytic median | test-only |
| A-43 | A minor | handover period-END test + IAAFT `computed`/`computed_nonconverged` over RP/XC/VC/COH | test-only |
| A-54 | A nit | `_script_population.coordination_corpus_drivers()` single-sources the derived driver population | test-only |
| A-19 | A minor | `samples_unobserved`/`samples_stationary`/`samples_below_min_players` populated + conservation test | report-only |
| B m1 | B minor | `_absorb_handovers` absorbs a multi-step excursion that first returns to its level within tolerance | YES (segmentation) |
| B m5 | B minor | D1 pass-b combine expects every provider in the corpus (`_corpus_providers`), not the `--providers` subset | no |
| B m6 | B minor | `derivation.json` occlusion block records `n_calibrated` + `width_source` (calibrated/fallback_default) | artifact only |
| B m9 (part) | B minor | NOTICE signal-prep/spread/surrogate attributions (Moura 2016/2013, Folgado 2014, Duarte 2013, Richardson 2012) | doc only |

Earlier batches (blockers A-02…A-05; majors A-06…A-17, R2-1/2-3/2-4; the minors/consider/B-items closed in earlier
batches) are likewise done + tested. **Authoritative record = the ADR item-3 corrected-output list + the working-tree
diff (in-bar); the external `scratchpad/merge/round2_triage.md` ledger is a working aid only** (review R-1; consistent
with Part G). The diff is the evidence; the ADR is the canonical statement of every behaviour change.

## Part B — remaining bounded findings (to implement, test-first)

Each is small, independent of A-09 unless noted. Approach is RED→GREEN with a both-sides test; corrected-output
impact is stated and appended to §F / the ADR.

- **B m8 — thin artifact tests.** (test + tiny source) Assert `corpus_visibility` on `derivation.json`,
  `metrics.json`, `numerics_noflip.json` (today only `calibration.json`); add the `occlusion-cal` pass to D1's
  `stage_seconds`; replace the `_CLEAN` dict in `test_d1_reduce_hands_its_artifact_to_d2` with a real clean-tree
  provenance. No shipping-value change.
- **B m9 (glossary part) — Mardia & Jupp attribution.** Add `_A_TF58_MARDIA` and set the three circular RP entries
  (`coord_rp_mean_deg`, `coord_rp_resultant_length`, `coord_rp_circ_sd_deg`) to `Bourbousson … ; Mardia & Jupp (2000)`;
  update `_expected_attribution` for exactly those three. Doc/attribution only. (NOTICE part already landed.)
  *Note:* interacts with the glossary-gate widening (A-38) folded into A-09 §C.6 — implement there if A-09 lands first,
  else here with a targeted gate update.
- **B m3 — cluster null `segment_too_short` keys on CONTRIBUTING player runs (owner-RATIFIED rule-3 conformance fix,
  2026-10-04; moved out of owner-decision framing).** Rule 3's "contributing" governs the whole clause; "(or player
  run, for cluster)" only names the cluster's unit, not a looser threshold. Route the cluster null through the SAME
  `_segments_holding(segments, rows)` helper the pair families use, with `rows` = the usable cluster-sample indices the
  `rho_group` statistic reads (survive the usable/`min_players` mask, inside `[s,e)`) — so the two paths cannot drift.
  The shift bound still applies per contributing run (a contributing run shorter than `2τ+1` → `segment_too_short`).
  Loosening direction only: a touching run that contributes no usable sample moves `segment_too_short → computed`; a run
  with no usable sample still cannot produce a null, so the family-level too-short guard still fires when nothing is
  scoreable. Tests (both sides, RED first): a planted run touching `[s,e)` but below `min_players` throughout → now
  `computed` (was `segment_too_short`); a contributing run shorter than `2τ+1` → still `segment_too_short`; a mutant
  reverting to "any touching run" turns the first RED. Corrected-output: shipped cluster-null rows move
  `segment_too_short → computed` — append to ADR item-3 + §F (signal-numerics, both no-flip legs share it); regenerate
  the as-built reference at the commit-1 tree. **The implementation must also AMEND the now-contradicting ADR prose
  (review R-2): the "keeps touching" / "any run touching the window" wording (ADR ~:446) and the R1 cluster ruling
  (ADR ~:289) are rewritten to the ratified "contributing" rule — not left standing beside a new item-3 entry, or the
  ADR self-contradicts.** If instead "touching" were kept, that needs an explicit recorded ruling + a proof that every
  touching run holds a usable cluster sample (touching ≡ contributing for the collective); absent that proof, "keep
  touching" is a knowing divergence from rule 3 and the pair families — a finding.
- **B m4 / minor 4 — `StoreConfig(objective_id=…)` type-ignores: determinism + wrong ignore form (owner 2026-10-04).**
  The cross-round 0-vs-2-errors split is not a reviewer error — each run is correct for the `ruthless-efficiency` the
  floating `>=0.7.0` resolved (newer stub carries `objective_id` → 0; exactly 0.7.0 omits it → 2). **Root cause = the
  floating pin** makes the CI type-check non-deterministic (a future ruthless release flips 0↔2 with no tree change).
  Steps: (1) **measure** — fresh-install `ruthless-efficiency[optuna]>=0.7.0` as CI does, record the resolved version,
  inspect whether its `StoreConfig` stub carries `objective_id`; (2) **fix the ignore FORM regardless of version** —
  `# type: ignore[call-arg]` is a MYPY code; pyright treats bare `# type: ignore` as suppress-ALL for the line (can hide
  an unrelated real error), so any kept ignore becomes `# pyright: ignore[reportCallIssue]` (that diagnostic only) with
  the exact version + stub-gap reason, at EVERY `StoreConfig(objective_id=)` site (calibrate_coordination.py,
  train_xcross_attempt.py, train_xshot_occurrence.py); report the stub omission upstream to ruthless; (3) make the end
  state DETERMINISTIC — remove the ignores if the pinned stub is complete, keep (corrected form) if not. **Owner rules
  the exact pin** — bounding `ruthless-efficiency` is a repo-wide dep-config change, not a TF-58-local one; surface the
  floating-pin hazard + proposed bound, owner decides the exact pin (the repo pins deps exactly elsewhere for this
  reproducibility reason). Lint-only; no corrected-output, no reference regen. NIT severity (objective_id is passed
  correctly; runtime unaffected).
- **B m15 — numba-gating population is a name list.** Broaden `test_numba_cache_gating.py` population detection to
  catch aliased imports (`from numba import njit as nb`) and `vectorize`/`guvectorize`/`cfunc` with `cache=True`; the
  current population is complete, so this is hardening, test-only. (Owner note: confirm scope — see §D.)
- **no-flip exclusions gate (owner-RATIFIED 2026-10-04; was a Part-D nit, now a planned gate fix).** A PASS requires
  `scored ∪ declared_excluded == corpus`, where `declared_excluded` is a DECLARED INPUT (committed, or the
  `--corpus-json` companion), NOT the run's own `.excluded.json` output — reading back the run's own exclusions is the
  R2-1 circularity. The gate FAILS if the run's actual exclusions ≠ the declared set; an undeclared exclusion is a
  regression. Each exclusion carries a reason from a CLOSED vocabulary (e.g. `no_tracking`, `empty_after_filter`, an
  owner-ruled token), not free text, so "unexpected" is machine-decidable. Promote `n_excluded` + each excluded key +
  reason into the PRINTED verdict next to `no_flip`. Tests (RED first): an undeclared exclusion FAILS; the same
  exclusion declared with a reason PASSES; actual ≠ declared FAILS; the mutant dropping the allow-list check turns the
  undeclared case green. Gate-report change only — no shipped metric moves, no corrected-output, no reference regen.
  **Owner input still needed (small):** the reason vocabulary tokens + the initial declared-exclusion set for the
  corpus (surfaced, not decided here).
- **A-56 (fallback_reason part) — `fallback_reason="gate not cleared"` also written when nothing moved**
  (`calibrate_coordination.py`). Distinguish "gate not cleared" from "nothing moved" in the recorded reason; small
  source + test. (The spec §8.1 K=1/K rationale amendment already landed.)
- **A-53 (remaining) — statistics corrections.** Split-half reliability reported without Spearman–Brown and unlabelled
  as half-length; sign tests count ties as failures; TOST reports "not equivalent" when all differences are identical;
  H7 `n_surrogates=200` vs the 199 convention. Each a focused source+test. **Overlaps A-09** (the reliability block
  reworks split-half); sequence A-53 inside / after A-09 to avoid double work — see §C.6.
- **nits:**
  - `resample_frames` raises `TypeError` on pandas nullable `Float32`/`Float64` input (`np.ndarray.astype("Float32")`,
    `_butterworth.py`). Accept nullable float (cast via `.to_numpy(dtype=float)`); test both dtypes.
  - `visibility_preflight` records nothing when all three probes of a provider fail to load (print-only). Record the
    all-fail outcome in the worker manifest; test.
  - ADR float-rounding figure "up to 1.7e-6" → the tested-tree max 1.4788e-6 (next 4.04e-11). Doc.
  - `pyproject.toml` "(TF-58, ADR-DRAFT)" → final ADR number. **Commit-time** (with the ADR rename, §E).
  - `no_flip` does not require `n_excluded == 0` (recorded under `population`). **Owner glance** — see §D.

## Part C — A-09: per-construct reliability and occlusion (SPEC + PLAN)

A-09 is the one block large enough to need its own spec. It implements goal 5 (`spec:262`, "reliability and
poolability for **every metric**"), §8.5 ("per metric and provider"), plan Task 23, as narrowed by the owner's
batch-3 rulings. The author's "one representative metric per family" is the DEFECT this replaces.

**A-09 CHANGES THE SPEC, so the spec is amended FIRST (review P-1, BLOCKING).** Today §8.5 / goal 5 specify
per-metric **linear** ICC(1) / split-half / Type-II. This section defines a different contract (per-construct grain,
rotation-invariant circular reliability, a power-verdict/"unmeasurable" terminal state, per-(match,entity) unit keying
with two honesty lines, per-construct occlusion). Because the spec wins over the plan, these requirements must land in
`docs/superpowers/specs/2026-09-26-tf58-team-coordination-design.md` (§8.5 + goal 5 + §9 as needed) **before** any A-09
code and **before** the as-built reference is regenerated (C.8.1, C.8.9). This is a genuine strengthening — a plain ICC
on a circular mean is origin-dependent, which is a latent §8.5 defect — not a plan-level override of the spec.

### C.1 Grain — per construct (owner ruling)

The reliability/poolability/decile/split-mode analyses (D3) **and** D1's occlusion comparison are keyed **per
construct**, not per family. A construct = a metric column × the keys that define *what* it measures:

- pair families: `level` + `signal_a` + `signal_b` + `axis`;
- spectral: `signal`;
- cluster / team-sync / RSI: `axis`;
- phase-row metrics also `phase_index`.

This is ~2000 cells in `metrics.json`. `report.md` carries a **descriptive** per-column summary (median + range of ICC
across that column's constructs) — explicitly NOT presented as "the column's ICC".

### C.2 Every cell carries power honesty

Each cell records `n` (groups/obs), a CI, and an **explicit power verdict**. An underpowered cell is terminal
**"unmeasurable"**, NEVER pooled up a level to rescue power.

The power verdict's thresholds are **named, pre-registered constants** — fixed in the C.8.1 spec amendment BEFORE the
corpus is scored, so `min` is not a post-hoc free parameter across ~2000 cells (review PLAN-02, matching the occlusion
side's `OCCLUSION_MIN_MATCHES_PER_BIN`):
- `RELIABILITY_MIN_N_GROUPS` — a cell with fewer measured groups is `unmeasurable` (`unmeasurable_reason="n<min"`);
- `RELIABILITY_MAX_CI_HALFWIDTH` — a cell whose reliability CI half-width exceeds this is `unmeasurable`
  (`"ci_too_wide"`), so a cell that is nominally powered but hopelessly imprecise is still terminal;
- `CIRCULAR_RELIABILITY_MIN_RBAR` (C.3) — a circular-mean construct whose mean resultant length is below this floor is
  `unmeasurable` (`"Rbar->0"`), because rotation-invariant circular reliability is undefined as concentration → 0.

The exact values are set in the spec amendment (owner-ratifiable there); the plan's contract is that they are named and
pre-registered, and the `unmeasurable_reason` tokens in the C.7 schema are exactly `{n<min, ci_too_wide, Rbar->0}`.

### C.3 Circular constructs

`coord_rp_mean_deg`, `coord_vc_mean_angle_deg`, `coord_phi_mean_deg` are circular throughout:

- D1 occlusion error = wrapped angular distance; between-match spread = circular SD of per-match circular means;
  deciles report circular median + spread; the 0.5·SD occlusion bar uses the **circular** SD.
- D3 binding reliability = **rotation-invariant circular reliability** `1 − (within-match circ var / total circ var)`
  via mean resultant lengths — NOT a cos/sin-component ICC (origin-dependent). cos & sin ICCs are kept as
  **diagnostics**, with the origin pinned in `derivation.json`. Low concentration (R̄ → 0) → unmeasurable, report R̄.
- The dispersion columns (`coord_rp_circ_sd_deg`, `coord_vc_angle_variability_deg`, `coord_phi_sd_deg`) are magnitudes
  → stay **linear**.
- Circular columns are declared in a **gated registry** with an anti-rot meta-test.

New kernel: a rotation-invariant circular-reliability function (R̄-based within/total circular variance), unit-tested
against a hand/analytic case and a rotation-invariance property (adding a constant phase to every value leaves the
reliability unchanged — the whole point).

### C.4 Player / pair units

Group on the measured **entity**: cluster-player = player; dyad = unordered player pair (canonical via `id_compat`,
gated); team-level = team. The unit is declared per construct in `metrics.json`. Keys are **per-(match, entity)**:
player ids are match-local on anon corpora, so retest axis = halves, group = `(match, player)` / `(match, unordered
pair)`, 2 obs each. **Never** key a global player id across matches on anon data. Report `n_groups`/obs/CI;
underpowered → unmeasurable, never pooled to team.

Two **report-honesty lines** (both in `report.md` and `metrics.json` provenance):
1. across-halves = within-match split-half (shared opponent/setup) = internal consistency = an **upper bound** on true
   match-to-match reliability;
2. cross-match player reliability is **unmeasurable** on anon corpora (no roster linkage); stable-roster providers are
   noted as a future extension.

### C.5 D1 per-construct occlusion (consumed per family; owner ruling on the C.8.6 grain, 2026-10-04)

`min_observed_fraction` is **consumed per family** — the library's detection gate is inherently per-family
(`params.min_observed_fraction[family]`, one float per method family; 5 sites in `_compute.py`), and the public
`CoordinationParams.min_observed_fraction` stays `Mapping[family → float]` (no seam change). The consumed threshold is the
**MAX over the family's governed constructs** of each construct's smallest qualifying share (**fail-closed** — MAX ≥
every construct's own share, so each is gated at least as strictly as its own curve requires; correctness identical to a
per-construct gate, only retention differs). A construct's qualifying share is the smallest observed-fraction bin whose
median absolute error is at most `0.5·SD` under full observation, where the curve is **estimable**: by an OBJECTIVE
criterion (review P-8) every occlusion decile bin it needs carries at least `OCCLUSION_MIN_MATCHES_PER_BIN` matches (a
named constant, from the GS+IDSSE occlusion corpus, recorded in `derivation.json`) AND the curve has a finite, unique
crossing of its `0.5·SD` bar (circular SD for the circular-mean constructs, C.3). A construct whose curve never clears
the bar below 1.0 → the family falls back to **1.0 (full-observation only)**, reported as a **FINDING**, never silently
(such a family yields ~no SkillCorner rows on broadcast ≈ 60% detection — state it, do not lower the bar to manufacture
rows).

The full **per-construct analysis is recorded** (not consumed): `derivation.json` holds, per construct, the curve + n/CI
per bin + the estimable/non-estimable verdict + the binding construct, **plus an over-restriction diagnostic** — per
family, how much more strictly the family MAX gates than each construct's own share, and any family driven to 1.0 by a
single construct. That diagnostic is the owner's decision input for the **per-construct-consumption follow-up**:
an owner-gated, ADR-costed change (a private per-construct threshold table behind the gate, leaving the public dataclass
a family-MAX summary), taken ONLY if the diagnostic shows the retention loss is material — it changes no correctness,
since MAX is already fail-closed. True per-construct *consumption* this cycle (widening the public dataclass to ~1000s of
per-construct entries, reworking 5 gate sites, generated-module bloat, byte-identity churn) is **out of scope** by that
ruling.

These per-construct curves inherit the synthetic-FOV occlusion model; its ratification (marginal detected-fraction
check against real SkillCorner + the scoped spatial-fidelity follow-up) is the Part-D A-08 item, and one
construct-validity "what it does NOT measure" line covers both the family thresholds and these A-09 curves.

### C.6 Findings folded into A-09

- **A-34** D2 per-fold vector misalignment → folds keyed by index (NaN kept), tested.
- **A-35** D2 selection and D3 reporting use the **same** reliability definition (review P-2: "or disclosed" is too
  soft — D2 must not select on a linear ICC the D3 report then presents as circular reliability). Since A-09 makes the
  circular-mean constructs' binding reliability rotation-invariant (C.3), D2's objective reads that same definition for
  those constructs; any residual difference is a ruled reconciliation, not an optional disclosure. The D2-gate
  population dependence of H1–H7 is also stated in `metrics.json`/`report.md`.
- **A-36** `type_ii_slope` relabelled **SMA/RMA** (or MA implemented), value pinned.
- **A-38 / B m10** glossary gate widened to goal 4: coverage entries carry scale + direction; a §8.5 allow-list; "no
  invented value" check; `coord_median_freq_cpm` definition corrected ("whole DC-excluded spectrum", not "within the
  analysis band"). (Absorbs B m9-glossary §B if not already landed.) **Closes r2 minor 10(c) (review P-7):**
  `_expected_attribution` must stop returning `None` for the cluster / SampEn / team-sync / `phi` / `rho` entries —
  map each to its source (Richardson+Frank for cluster/phi/rho, Richman+Duarte for SampEn/Cross-SampEn, Duarte for the
  team-team Pearson r) so the attribution gate actually tests them.
- **B m7** D2 CV folds keyed on the provider-qualified `join_key`, not bare `match_id`.
- **A-53 (reliability parts)** split-half Spearman–Brown + half-length labelling land here, as the reliability block is
  rebuilt; the sign-test/TOST/H7 parts stay in §B unless they touch the same code.

### C.7 Outputs — `metrics.json` schema (review P-8: written here, not "on acceptance")

Top-level keys: `schema_version` (str), `run_commit` / `run_tree_dirty` / `run_tree_state` (provenance),
`run_tree_hash` (A-55), `corpus_visibility` (ADR-038), `input_contract`, `params` (source + sha256 of the D1/D2
artifacts, M-5), `population`, `honesty` (the two C.4 lines as strings), and `constructs` — a LIST of per-construct
cells. Each cell:

```
{
  "column":        "coord_rp_resultant_length",     # the metric column
  "construct_key": {"level":"team_team","signal_a":"centroid_x","signal_b":"centroid_x","axis":"x"},  # C.1 keys
  "unit":          "team",                           # measured entity (C.4): team | player | unordered_pair
  "kind":          "linear",                         # linear | circular   (C.3)
  "reliability": {                                   # the BINDING definition (circular constructs: rotation-invariant)
      "value": 0.63, "ci": [0.48, 0.75], "estimator": "icc1" | "circular_reliability",
      "n_groups": 71, "n_obs": 142,
      "power": "measured" | "unmeasurable",          # C.2: underpowered -> "unmeasurable", value may be null
      "unmeasurable_reason": null | "n<min" | "ci_too_wide" | "Rbar->0"   # C.2 pre-registered thresholds
  },
  "diagnostics": { "icc_cos": 0.6, "icc_sin": 0.58, "origin_deg": 0.0 },  # circular only; else {}
  "poolability": { ... same shape ... },
  "split_mode":  { "half_length_s": 2700.0, "spearman_brown": true, "value": 0.7, ... },  # A-53 labelled
  "deciles":     [ {"bin": 0.6, "n": 12, "median": ..., "spread": ...}, ... ]  # decile stratification
}
```

Phase-row metrics add `"phase_index"` to `construct_key`. `report.md` carries, per column, a DESCRIPTIVE summary
(median + range of `reliability.value` across that column's constructs, with the count of `unmeasurable` cells) —
never presented as "the column's ICC" — plus the two §C.4 honesty lines verbatim. The metric-family output contracts
(`metric_contracts`) register any new emitted columns, or the ADR-098 gate fails.

### C.8 Task breakdown (test-first)

1. **Amend the SPEC first (blocking, review P-1).** The spec wins over the plan, and §8.5 / goal 5 today specify
   per-metric **linear** ICC(1) / split-half / Type-II; **§8.4 defines D2's ICC(1) selection objective** — amend it too
   (review R-3), since P-2 requires D2 to select on the SAME reliability definition D3 reports. Before any code or any
   reference regeneration, amend the spec (§8.4 + §8.5 + goal 5 + §9 as needed) to the per-construct contract this
   section defines: the per-construct grain (C.1), the per-cell power verdict and
   "unmeasurable" terminal state (C.2), the rotation-invariant **circular** reliability for the three circular-mean
   constructs with cos/sin kept only as diagnostics (C.3), the per-(match,entity) unit keying + the two honesty lines
   (C.4), and the per-construct (or MAX-family) occlusion threshold (C.5). Record the amendment in the spec's change
   log and cross-reference the owner's batch-3 ruling. Without this, a spec-wins round-3 review flags the whole block
   and C.8.9 would encode an un-specced oracle. (The circular kernel fixes a real §8.5 defect — a plain ICC on a
   circular mean is origin-dependent — so this is a strengthening the spec must carry, not an override of it.)
2. Circular-reliability kernel + gated circular registry + anti-rot test. (new module under `coordination/` or
   `calibration/`)
3. Per-construct key derivation helper (one source for D1 + D3 grain).
4. D3 reliability/poolability/decile/split-mode rebuilt per construct; power verdict per cell.
5. Player/pair unit keying + honesty lines.
6. D1 per-construct occlusion curves + per-construct (or MAX-family) `min_observed_fraction`; `derivation.json`
   records binding construct + curves.
7. A-34/A-35/A-36/A-38/m7 folded in with their tests.
8. `metrics.json` schema (per §C.7, written in this plan before implementation) + `report.md` summaries;
   `metric_contracts` registration.
9. Regenerate the as-built reference numerics at the commit-1 tree (owner ruling) — AFTER the spec amendment (C.8.1),
   so the oracle encodes the specced contract.

### C.9 Corrected-output

A-09 moves the D3 reliability/poolability numbers wholesale (new definitions + grain) and D1's `min_observed_fraction`
(per-construct). These are NOT in the no-flip signal gate (that compares signal numerics), but they change
`metrics.json` / `derivation.json` and the committed as-built reference → regenerate per C.8.9, AFTER the spec
amendment (C.8.1).

## Part D — owner-decision items (surface + recommendation)

These are not mine to pre-decide; each carries a recommendation for the owner / the two sessions to rule on. Flagged
in the ledger; none blocks the bounded work.

- **A-49(a) — restart dead-interval start: KEEP `t[i-1]` + document (owner-RATIFIED 2026-10-04, with an evidence
  refinement).** `t[i-1]` is the only non-arbitrary timestamp (last instant the ball is known alive; SPADL has no
  end_time); the 25 s split threshold absorbs the prior-action-duration slack in the conservative direction; consistent
  with the goal case's symmetric `[goal, kickoff]`. No behaviour change, no corrected-output. **Refinement (owner):**
  quantify the bound rather than assert it — the D20 stoppage leg already measures event-derived stoppages (>25 s)
  against true `ball_state` on GS+IDSSE; add a reported FALSE-POSITIVE rate = fraction of restart intervals where
  `t_restart − t[i-1] > 25 s` but the true `ball_state` dead span is ≤ 25 s (the slack-only splits). Document the
  limitation precisely in `_event_intervals` + the ADR (SPADL model: no end_time), naming where it bites (restart
  intervals whose true start is within one prior-action duration of the 25 s boundary). Doc + D20 measurement only."
- **A-39 interim-band-for-floor: MEASURE the gap, then ratify (owner 2026-10-04).** A-39 landed with the spectral
  min-length floor keyed to the INTERIM band_low (pass-B derives the band; default `CoordinationParams` band is used).
  Before ratifying, measure (cheap, no re-run — pure recompute over the already-cached pass-B shards + the two band_low
  values, interim-default vs final-derived, on GS+IDSSE): **(a)** segments whose floor verdict differs between the two
  bands (`len` vs `min_spectral_samples(fs, band_low)` for each); **(b)** of those, how many actually entered/left a
  window's spectral-median contributor set; **(c)** the resulting max spectral-median delta (the shipped-number move;
  (a) alone overstates it). Flag the ANTI-CONSERVATIVE direction explicitly: `min_spectral_samples ∝ 1/band_low`, so if
  the derived band_low < the interim default, the final floor is HIGHER and the interim floor ADMITS a segment the final
  band would reject (a too-short segment contributing to the median). **Decision rule:** (b) = 0 → ratify the interim
  floor, record the measurement in the ADR as the evidence, no further corrected-output; (b) > 0 with the
  anti-conservative direction present → reopen (fixed-point re-derivation, or an owner-approved acceptance of the named
  affected segments) — a NEW corrected-output + reference regeneration. Runs on MEDIA-PC with the real D1 cached shards
  (needs a corpus D1 run); attach the three numbers to the ratification. The measurement itself changes no behaviour."
- **B m11 — verified in-bar (review P-5), recording gap only.** The owner's batch-2 PROCEED (tau=40 mechanism probe,
  not an endorsed value) is implemented and tested IN-REPO:
  `tests/coordination/test_liveness.py::test_idsse_half_xc_null_hinges_on_the_interim_min_shift` parametrizes
  `min_shift_s=60.0 → segment_too_short` vs `40.0 → computed`, and the idsse_half liveness smoke pins the three
  not-live columns to the interim `min_shift_s`. *Rec:* close — the "unverified" status was because the test lives in
  the repo, not in the external ledger the sessions could not see.
- **B m13 — plan↔test name reconciliation (sign-off, not a branching decision).** Several plan Task-17 test names were
  renamed to equivalents; the window-order determinism gap is already closed
  (`test_determinism_is_independent_of_window_order`, landed this session). Planned in Part B with the m12 file-map
  refresh: produce an explicit mapping (plan name → actual test name → what it covers), update the plan to the actual
  names. The owner reviews the mapping in round-3b and signs off.
- **A-19 grains** (if revisited) — the three sample-counter grains chosen (unobserved at signal build; stationary per
  scored-diff pair; below-min per team/axis). *Rec:* accept as implemented.
- **A-08 synthetic-FOV occlusion model — PROCEED with a mandatory marginal validation first (owner batch-2 "flag before
  DGX" + 2026-10-04 refinement).** The detection construct / D1 occlusion leg (and the A-09 per-construct curves that
  inherit it) model broadcast occlusion with a synthetic FOV mask (`fov_mask`/`simulate_broadcast_occlusion`) on GS+IDSSE.
  Do NOT block the cycle (dropping or deferring all validation) and do NOT drop the §8.2 D9 leg. Instead:
  1. **Measure the MARGINAL now (no labelled ground truth needed):** real SkillCorner per-(window, side) detected-fraction
     distribution (from its real `visibility` flag — the source of the ~60% figure) vs the synthetic mask's distribution
     on GS+IDSSE; report per-bin overlap + divergence. Cheap, no DGX, no corpus driver if a representative SkillCorner
     sample suffices (local/MEDIA-PC).
  2. **Test the fail-closed direction explicitly:** `min_observed_fraction` bins occlusion error by detected fraction, so
     "conservative (over- not under-restriction)" is only true if the synthetic mask is at least as aggressive as real
     occlusion in every occupied bin. FLAG any bin where synthetic is LAXER than real — that is the one case that breaks
     the conservative framing and must be seen before ratifying. The 1.0 full-observation backstop (A-09 §C.5) bounds the
     downside: a dubious curve falls to full-observation-only as a FINDING, not silently lax.
  3. **Caveat precisely:** after the marginal check, the genuinely-unvalidated part is the SPATIAL/joint structure (which
     players drop together), which needs labelled ground truth → a SCOPED, owner-approved follow-up **recorded in the
     ADR** (not self-parked on a TODO — the "flag before DGX" ruling is the sign-off gate). The construct-validity "what
     it does NOT measure" line states: thresholds rest on a synthetic FOV whose MARGINAL detected-fraction was checked
     against real SkillCorner; its spatial fidelity is not validated.
  Attach the marginal number to the ratification. No corrected-output from the measurement itself (it ratifies or
  reopens the thresholds; only a recalibration would move numbers + need reference regen). A-09 §C.5 carries the same
  caveat (one construct-validity line covers the family thresholds and the A-09 curves).

## Part E — end-of-cycle sequence (unchanged from the original plan's Task 27, restated)

1. Full coordination + driver/gate/seam suite GREEN locally (sanity).
2. **MEDIA-PC** full suite (both CI legs, `OMP_NUM_THREADS=4`) + a fresh 50-match no-flip. The numerics gate now needs
   the D1/D2 artifacts (`--derivation`/`--calibration`) or `--in-package-params` (recorded dev run). Delegated to
   MEDIA-PC (local box crashes on memory pressure; DGX in use).
3. ADR corrected-output list finalised; ADR file **renamed** `ADR-DRAFT-tf58-team-coordination.md` →
   `ADR-111-team-coordination.md` (DONE; ADR-110 on `main` = raw-primitives boundary); `pyproject.toml` ADR ref =
   ADR-111 already.
4. **Round-3 review:** the two sessions `/review-plan` THIS document, then `/review-impl` the working-tree diff; fix
   any CONFIRMED findings.
5. **Commit-1 diff STOP** — show the full diff / file list; wait for explicit owner approval. No commit before it.
6. DGX authoritative full-corpus no-flip on the approved commit-1 tree (owner-coordinated; after the other session's
   weights-only merge completes).

## Part F — consolidated corrected-output ledger

**The ADR's item-3 list is the CANONICAL, complete record of every behaviour change** (review P-3 — this §F is a
plan-side INDEX, not the authoritative "every change" list; read the ADR for completeness). Indexed here, for the
no-flip gate's reference regeneration and the as-built oracle.

Signal-numerics changes (in the no-flip gate's scope): A-06 (spectral median bin rule), A-07 (bridge-then-filter),
A-08 (detection gate on all families + `coord_detected_share`), A-20 (SampEn source split / degenerate tokens),
A-21 (cluster single-sample → too_short), A-22 (window `n_phases`), A-23 (pair order A=attacking), A-24
(goal_end_unresolved on back-line signals), A-25 (SampEn within-run + φ unwrap), A-30 (no-row windows), A-45
(phase-valid run-first), A-48 (exact pre-warped Butterworth, coordination opt-in), A-51 (coupling-angle series
admission), B m1 (multi-step handover), B m3 (cluster null contributing-run rule: `segment_too_short → computed` on
non-contributing touching runs) — **and the earlier-batch movers the ADR item-3 list carries that were omitted
from the first draft (review P-3): the long-stoppage player-run split (`_period_stoppages`), the possession-spectrum
per-window×segment slice, the contributing-segment surrogates (`_segments_holding`), and A-52 (`welch_segment_s` /
`RESIDUAL_GRID` range).** Both legs of the no-flip share all of these, so the gate stays balanced; the committed
as-built reference must be regenerated at the commit-1 tree.

Non-signal (artifact/report/gate) changes: A-14 (D1 noise floor), A-17 (IAAFT refusal, no value move), A-19 (report
counters), A-33 (store id), A-39 (D1 band/min_shift), A-55 (no-flip verdict), B m6 (occlusion `width_source`),
A-09 (D3 reliability + D1 per-construct `min_observed_fraction` — the large one; also moves `metrics.json` wholesale).

## Part G — owner rulings, transcribed in-bar (review P-4)

The owner's AskUserQuestion rulings (2026-10-04) governed the scope decisions this cycle. They lived only in the
session-external triage ledger, which a review in an isolated clone cannot see (review P-4). Transcribed here so they
are in-repo and verifiable; the per-finding CLOSURE record is the ADR item-3 corrected-output list + the working-tree
diff (the external `scratchpad/merge/round2_triage.md` remains a working aid only).

**Batch 1.**
- A-27: spec interval `[τ, N−τ]` CLOSED; REFUSE thin nulls (fewer distinct joint draws than `n_surrogates` →
  `segment_too_short`); report counts of such refusals + NaN draws.
- A-20: SEPARATE SampEn source columns (`coord_cluster_sampen_source`, `coord_cluster_player_sampen_source`,
  `coord_team_sync_sampen_source`); spec 7.12 amended; contracts + glossary.
- Spec fixes approved: A-44 (225° → 135°/315°), A-56 (§8.1 K rationale → 1/K resolution), A-47 (pad cap len−1 stated),
  A-48 (EXACT pre-warped Butterworth cutoff; behaviour change; §7.4 step 3 amended).

**Batch 2.**
- A-08 CONSTRUCT: ONE detection construct, D1 matches production. Decision variable = MIN over the row's sides, in
  production AND D1; ONE helper both call; cluster PLAYER rows test `min(team, player)`; share = RAW detected mask over
  the side's on-pitch samples in `[s,e)` (bridged/interpolated = not detected) → fully observed always 1.0; empty
  window → `too_short`; on-pitch-never-detected → `insufficient_detection`. New `coord_detected_share` on all 7 tables.
  Possession rows EXEMPT from the gate; D1 reports occlusion error on possession as evidence. Supersedes the 2026-09
  "both scored fully observed" occlusion ruling. FOV-mask-vs-real-occlusion caveat flagged for the owner before the DGX.
- CHILD ROWS (A-21 + pair-phase): degraded cluster-team window → one player row per roster member; degraded pair row →
  the window's `n_phases` phase rows; keys real, metrics NaN, token = parent's reason; emission reason-INDEPENDENT.
  ADR-042 conservation gate covers both. Regenerate the as-built reference at the commit-1 tree.
- A-22: NA = NO subdivision; the window's own `n_phases` governs; validate at entry (NA or int ≥ 2); nullable Int64.
- A-52: KEEP 400 s (shortest meeting resolution); amend spec 8.2; the ≥ 8-segment check recorded; K per row + chance
  level = surrogate baseline (no new column).
- B m11: PROCEED — at τ=60 the 3 rows' null is `segment_too_short`, at τ=40 `computed` + live; τ=40 is a MECHANISM
  PROBE, not an endorsed value; D3 reports the computed-null share per column × provider × window-kind × token
  (report-only). (Implemented; see Part D B m11.)

**Batch 3 (A-09 "every metric").** GRAIN per construct (D3 reliability/poolability/decile/split-mode AND D1 occlusion;
unit = metric column × the keys that define what is measured; ~2000 cells; `report.md` descriptive per-column summary,
never presented as the column's ICC; representative one-ICC-per-family is the DEFECT; every cell carries n + CI + an
explicit power verdict; underpowered = terminal "unmeasurable", never pooled). D1 THRESHOLD prefer per-construct
`min_observed_fraction`, else FAMILY = MAX over constructs; a construct never qualifying below 1.0 → family falls back
to 1.0, reported as a FINDING. ANGLES circular throughout (wrapped-distance error; rotation-invariant circular
reliability via mean resultant lengths; cos/sin kept as diagnostics; low R̄ → unmeasurable; dispersion columns stay
linear; gated circular registry + anti-rot). PLAYER UNIT: group on the measured entity; key per-(match, entity); never
a global player id across matches on anon data; two report-honesty lines (within-match split-half = upper bound;
cross-match player reliability unmeasurable on anon corpora). SCOPE NOTE: implement test-first; do NOT defer any part
without asking. (This batch is the source for Part C.)

**Batch 4 (2026-10-04, the six Part-D open-question rulings) — transcribed inline where each lands, cross-referenced here
(review CONSIDER):** Q1 B m3 → "contributing" (Part B); Q2 A-49(a) → keep `t[i-1]` + D20 false-positive measurement
(Part D); Q3 A-39 → measure the interim-vs-final floor gap, decision rule (Part D); Q4 exclusions → declared allow-list
as an INPUT + `scored ∪ declared_excluded == corpus` (Part B); Q5 minor-4 → measure the CI-resolved version + fix the
pyright ignore form + owner-ruled deterministic pin (Part B, "B m4"); Q6 A-08 FOV → proceed + mandatory marginal check
now + owner-approved spatial follow-up in the ADR (Part D + C.5). Each ruling's full text sits with its finding.

**Batch 5 (2026-10-04, the C.8.6 occlusion-grain ruling) — transcribed in §C.5 and the ADR:** `min_observed_fraction`
is consumed **per family** as the fail-closed MAX over the family's constructs' qualifying shares (Option 1); the public
`CoordinationParams.min_observed_fraction` stays `Mapping[family → float]` (no seam change, `_compute.py` untouched).
Three refinements: (1) amend §C.5 + the §8.2 spec text to family-MAX consumption [done]; (2) record the over-restriction
diagnostic in `derivation.json` as the owner's decision input [done — `occlusion.per_construct` + `over_restriction`];
(3) per-construct *consumption* is an owner-gated, ADR-costed follow-up (private per-construct table behind the gate, no
public-dataclass change), taken only if the diagnostic shows the retention loss is material [recorded in the ADR]. Full
per-construct consumption (widening the public dataclass) is OUT of scope this cycle.
