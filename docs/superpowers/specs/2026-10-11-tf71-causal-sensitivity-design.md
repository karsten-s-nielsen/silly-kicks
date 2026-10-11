# TF-71 — Unmeasured-confounding sensitivity for `silly_kicks.causal` (design)

**Status:** Option A (reimplemented-ZSB IPW-ATT) **adopted and APPROVED.** Spec reviews (through r4) and impl reviews (r1 + r2) by both independent reviewers (A + B) all APPROVE. The E-value RR+CI is on the IPW estimand (owner-ruled SENS-SPEC-05; matched RR a companion); the `ipw.R` oracle covers `_extrema_os_att` + AIPW via a test-only injection seam (SENS-SPEC-06/07); the NO-GO bars are pinned in-spec (agreement = 1 A&I SE; ESS ≥ 0.10·n, max weight ≤ 0.10); the point-vs-curve caveat is sharpened. Owner ruled the derivative-work posture **MIT-clean** (2026-10-11): an independent reimplementation of a published algorithm, with `bootsens` an out-of-process oracle (never vendored/imported) — the ADR-107/108 native-reimplementation precedent. **Awaiting only the owner's explicit per-commit yes.** SENS-SPEC-04 resolved (§Scope boundary).
**TODO row:** TF-71 (On Deck). **This cycle ships ADR-117** (the decision record). **ADR precedent:** ADR-015 (causal-validation port), ADR-037 (TF-19 as the second consumer), ADR-056 (declared input contracts), ADR-009 (reported-not-gated research), ADR-107/108 (native-reimplementation precedent), ADR-110 (raw-primitives boundary).

## Executive summary (for the external reviewer)

`silly_kicks.causal` (`causal/matching.py`, ADR-015) reports a propensity-matched ATT/ATNT, an Abadie–Imbens SE, SMD balance, and a permuted-block placebo band. It reports **nothing about how strong an unmeasured confounder would have to be to explain an estimate away** — the robustness rung of the validity ladder. This cycle adds that, as a new pure sibling module `causal/sensitivity.py`, in two forms the owner pre-ruled on 2026-10-03:

1. **The E-value** (VanderWeele & Ding 2017) — the minimum strength of association, on the risk-ratio scale, that an unmeasured confounder must have with **both** treatment and outcome to fully explain the observed effect (point + the E-value for the confidence limit nearest the null). The E-value is a closed form of a risk ratio; to keep its **confidence** interval valid, both the RR point and its CI are computed on the **Hájek-IPW estimand** (weighted outcome rates, same propensity) with ZSB's valid percentile bootstrap — NOT on the matched set, whose resample-and-rematch bootstrap A&I-2008 shows is invalid. The **matched** risk ratio is reported as a companion (agreement check), mirroring the matched-headline / IPW-inference decoupling used for the MSM.

2. **The marginal sensitivity model** (Zhao, Small & Bhattacharya 2019) — **computed faithfully on a Hájek-IPW ATT**, not on the matching estimator. ZSB prove the MSM + percentile-bootstrap for the Hájek-IPW estimator; the reference implementation `bootsens/R/ipw.R` already exposes `estimand = "att"`. Running the MSM on a Hájek-IPW ATT built from the **same fitted propensity** the matcher uses **inherits the theorem** — no derived tilt, no invented constants, no matching bootstrap. (The alternative — tilting the matcher's `K_j` weights — is an unproven off-label construction for a 1:1-NN-with-replacement, no-caliper matcher; Rosenbaum's Γ also assumes disjoint pairs and does not apply. Option A puts the MSM on the estimator the model is built for. Otsu & Rai 2017 is a **wild bootstrap** for the *matched-ATT sampling variance* — not an MSM vehicle — and is **moot** here.)

**Decoupling, and the honesty boundary.** The **matched ATT stays the headline point estimate** (shipped, unchanged). The MSM rides a sibling Hájek-IPW ATT on the same propensity — the same ATT estimand under overlap. The MSM statement is formally "how strong must unmeasured confounding be to explain away the **IPW-ATT**"; it transfers to the matched ATT only insofar as the two agree. The result therefore reports **both** point estimates and their agreement gap; the arm is **NO-GO** (reported, not shipped as a robustness claim) if they diverge beyond a pre-registered band or if overlap fails (unstable Hájek weights).

The change is **additive and pure**: `CausalEstimate` is untouched, no new runtime dependency, nothing trains, nothing needs the DGX or a corpus run. The two consuming harnesses (`scripts/validate_xcross_causal.py`, `scripts/validate_xshot_causal.py`) are wired to call the new seam and emit the fields into their `metrics.json`; the committed memos are **not** regenerated this cycle (an owner-run corpus follow-up, surfaced as a new TODO row — the only owed item).

## Context: the existing estimator (read, not paraphrased)

`silly_kicks/causal/matching.py` (verified at `main` `02258e9`):

- `fit_propensity(X, Z, *, seed) -> (ps, coef)` — logistic propensity on **standardized** covariates, near-unregularized (`C=1e6`, lbfgs), scores clipped to `[1e-6, 1-1e-6]` (`:39-64`). This is the ONE propensity model both the matched ATT and the sibling IPW-ATT use.
- `propensity_match(ps, Z, *, target) -> dict[int, int]` — 1:1 NN on the score, **with replacement, ties allowed, no caliper** (`:67-88`).
- `estimate_att` / `estimate_atnt` → `CausalEstimate{estimate, se, balance, n_focal, matched}` (`:170-218`); `abadie_imbens_se` (`:139-167`), the naïve bootstrap biased under with-replacement matching (`:5-6`).
- `placebo_shift(...)` — a permuted-block alignment null (`:289-338`); NOT a confounding-sensitivity analysis.

Siblings already sit beside `matching.py`: `opportunities.py` (row builder), `power.py` (`att_power_curve`, `InjectionSpec`). A new `sensitivity.py` follows that shape. The opportunity builder's outcomes (`OpportunityConfig.outcome_*`) are **binary** windowed events — the E-value applies; the module guards the general case.

## Goals / non-goals

**Goals.** A pure, deterministic sensitivity layer: the E-value (binary outcome; IPW RR + matched-RR companion) and a reimplemented-ZSB MSM on a Hájek-IPW ATT (any outcome) with ZSB's own percentile bootstrap, reported with the matched-vs-IPW agreement gap; a doubly-robust (AIPW) variant offered; `bootsens/R/ipw.R` as a reference oracle; pre-registered gates; both consuming harnesses wired.

**Non-goals.** No change to `CausalEstimate` or any existing estimator; no derived matching-MSM / `K_j` tilt; no Rosenbaum Γ; no matching bootstrap (Otsu & Rai moot); no new runtime dependency; **no corpus run and no committed-memo regeneration** this cycle; no ranking or tactical blend (ADR-110).

## Design

### Module & public surface

New file `silly_kicks/causal/sensitivity.py`. Pure numpy/scipy + the existing `fit_propensity` (sklearn, fit-time only). Deterministic given `seed`. Exported from `causal/__init__.py`; registered in `tests/test_public_api_examples.py` `_PUBLIC_MODULE_FILES` (doctest on `sensitivity_analysis`; the frozen result dataclass in `_SKIP_SYMBOLS` per the `DetectionResult`/`OpportunityConfig` precedent).

```python
@dataclass(frozen=True)
class SensitivityResult:
    # --- E-value (None when the outcome is non-binary); IPW risk ratio carries the valid inference ---
    rr: float | None                  # Hajek-IPW risk ratio: mean(Y[A==1]) / ipw_E[Y(0)|A=1]
    rr_ci: tuple[float, float] | None # ZSB percentile-bootstrap CI of the IPW RR (resample + refit, no rematch)
    matched_rr: float | None          # the propensity_match RR, reported as a companion (agreement check)
    e_value_point: float | None
    e_value_ci: float | None          # E-value of the IPW-RR-CI limit nearest 1 (1.0 if the CI crosses 1)
    # --- marginal sensitivity model on the Hajek-IPW ATT (reimplemented ZSB) ---
    target: str                       # "att"
    ipw_att: float                    # the Hajek-IPW ATT point (gamma = 0)
    matched_att: float                # the shipped matched ATT (for the agreement diagnostic)
    agreement_gap: float              # |ipw_att - matched_att|
    agreement_ok: bool                # gap <= pre-registered band
    overlap_ok: bool                  # Hajek weights stable (ESS / effective-weight guard)
    msm_go: bool                      # agreement_ok AND overlap_ok  (else the MSM is reported NO-GO)
    reg_adjust: bool                  # False = Hajek-IPW; True = AIPW (doubly robust)
    gammas: tuple[float, ...]         # log-OR grid, ascending, gammas[0] == 0.0 (Gamma = e^gamma)
    msm_lo: tuple[float, ...]         # per-gamma extrema lower bound of the IPW-ATT (point)
    msm_hi: tuple[float, ...]         # per-gamma extrema upper bound (point)
    msm_ci_lo: tuple[float, ...]      # per-gamma percentile-bootstrap CI lower
    msm_ci_hi: tuple[float, ...]      # per-gamma percentile-bootstrap CI upper
    gamma_star: float                 # smallest gamma whose [ci_lo, ci_hi] includes 0; inf if none in-grid
    # --- provenance ---
    n_boot: int
    alpha: float
    seed: int
    outcome_is_binary: bool
    notes: tuple[str, ...]
```

Functions (the names mirror `ipw.R` so the oracle mapping is one-to-one):

- `_get_extrema(A, Y, fitted_prob, gamma, *, estimand_all) -> tuple[float, float]` — the closed-form IPW extrema, reimplementing `get.extrema`: `eg = exp(-logit(fitted_prob))`; restrict to `A != 0`; sort by `-Y`; `num_low/up = A*Y*(c + exp(∓gamma)*eg)`, `den_low/up = A*(c + exp(∓gamma)*eg)` with `c = 1` if `estimand_all` else `0`; `max = max(cumsum_up ⊕ rev_cumsum_low)/den`, `min` the mirror (the exact `c(0,cumsum(..))+c(rev(cumsum(rev(..))),0)` construction). Pure, no model fit.
- `_extrema_md(A, X, Y, gamma, *, estimand_all, reg_adjust, seed) -> tuple[float, float]` — reimplements `extrema.md`: fit the propensity (`fit_propensity`), optional outcome-regression adjustment on `A==1` for AIPW (`reg_adjust`), `_get_extrema(A, Y - Y_fitted, ...)` then add back the fitted mean.
- `_extrema_os_att(A, X, Y, gamma, *, reg_adjust, seed) -> tuple[float, float]` — reimplements `extrema.os(estimand="att")`: `mean(Y[A==1]) - reversed(_extrema_md(1 - A, X, Y, gamma, estimand_all=False, ...))` (the `rev` swaps min/max because subtracting an interval flips it).
- `_bootsens_att(A, X, Y, *, gamma, alpha, n_boot, reg_adjust, seed) -> tuple[float, float]` — reimplements `bootsens.os`: `n_boot` replicates, each `rng.choice(n, n, replace=True)`, **refit the propensity inside the resample** (ZSB's validity condition), percentile CI `(quantile(lower, alpha/2), quantile(upper, 1-alpha/2))`. One `default_rng(seed)`.
- `e_value(Z, X, Y, *, target, n_boot, alpha, seed) -> (e_point, e_ci, rr, rr_ci, matched_rr)` — **binary `Y` only**, raises `ValueError` otherwise. The **Hájek-IPW risk ratio** `rr = mean(Y[A==1]) / p0_cf`, where `p0_cf` is the IPW estimate of `E[Y(0)|A=1]` = the `gamma=0` point of `_extrema_md(1 - Z, X, Y, estimand_all=False, ...)` (reusing the ipw.R port + the shared `fit_propensity`). The RR CI is **ZSB's valid percentile bootstrap** (resample + refit, **no rematch**). The **matched RR** on the `propensity_match` set is returned as a companion (agreement check), not an inference basis. E-value closed form: `E = RR + sqrt(RR*(RR-1))` for `RR >= 1`, reciprocal for `RR < 1`; `e_ci` from the IPW-RR-CI limit nearest 1 (`1.0` if the CI crosses 1).
- `marginal_sensitivity(Z, X, Y, *, gammas, n_boot, alpha, reg_adjust, seed) -> SensitivityResult` — the reimplemented-ZSB MSM on the Hájek-IPW ATT (E-value fields `None`); computes `ipw_att` (`gamma=0`), the per-`gamma` extrema + bootstrap CI, `gamma_star`, the `matched_att`/agreement/overlap/`msm_go` diagnostics.
- `sensitivity_analysis(Z, X, Y, *, target="att", gammas=DEFAULT_GAMMAS, n_boot=DEFAULT_N_BOOT, alpha=0.05, reg_adjust=False, seed) -> SensitivityResult` — the doctested entry point: MSM always; E-value folded in when `Y` is binary, else `outcome_is_binary=False` and E-value fields `None`.

Pre-registered constants (frozen, named): `DEFAULT_GAMMAS` (log-OR; `0.0, 0.22, 0.41, 0.69, 1.10, 1.39, 1.79` ≈ `Γ ∈ {1, 1.25, 1.5, 2, 3, 4, 6}`), `DEFAULT_N_BOOT = 1000`, `DEFAULT_ALPHA = 0.05` (ipw.R defaults). **NO-GO bars, pinned HERE pre-numbers** (the plan may tighten, never loosen): `agreement_ok` iff `|matched_att - ipw_att| <= matched_se` (one Abadie–Imbens SE — self-scaling, principled, no arbitrary absolute threshold); `overlap_ok` iff the Hájek effective sample size `ESS = (Σw)²/Σw² >= OVERLAP_MIN_ESS_FRACTION · n_focal` with `OVERLAP_MIN_ESS_FRACTION = 0.10` AND no single normalized weight exceeds `OVERLAP_MAX_WEIGHT_SHARE = 0.10`.

### Gamma convention

`gamma = log Λ` (log odds ratio), `Γ = e^gamma`, matching `ipw.R`. `gammas[0] == 0.0` ⇒ `Γ = 1`. At `gamma = 0` the low/high weight factors coincide, so `_get_extrema` returns `min == max == the point IPW estimate` **by construction** — the "Γ=1 collapses to the unconfounded estimate" non-vacuity property is **automatic and exact under Option A**, not a derived-tilt claim to prove.

### E-value (VanderWeele & Ding 2017) — closed form on the IPW risk ratio

The E-value is a closed-form transform of a risk ratio; the only inference is the RR's CI. To keep that CI valid, the RR is the **Hájek-IPW risk ratio** `rr = mean(Y[A==1]) / p0_cf` (`p0_cf` = the `gamma=0` IPW estimate of `E[Y(0)|A=1]`, from the `ipw.R` port + the shared `fit_propensity`), and its CI is ZSB's valid percentile bootstrap (resample + refit, no rematch) — the same valid machinery as the MSM, on the same estimand. `E = RR + sqrt(RR*(RR-1))` (`RR >= 1`; reciprocal for `RR < 1`); `e_value_ci` from the CI limit nearest 1. The **matched** RR (`p_focal/p_comp` on the `propensity_match` set) is reported as a companion for the agreement check, NOT as the inference basis — mirroring the matched-headline / IPW-inference decoupling. (Why not the matched RR for inference: its resample-and-rematch bootstrap is the A&I-2008-invalid scheme; the IPW RR avoids it entirely.) Degenerate guards (`p0_cf == 0`, empty set) → `None` E-value fields + a `notes` entry, never a divide-by-zero.

### MSM on the Hájek-IPW ATT (reimplemented ZSB) — the reference-oracle algorithm

The Python functions above **reimplement the published ZSB algorithm** (the paper's `get.extrema` / `extrema.md` / `extrema.os` / `bootsens.os` recipe), **oracle-validated against** `bootsens/R/ipw.R`. No `bootsens` source is vendored or imported; the R package is used only as an out-of-process numeric oracle (GPL-3, dev-image only). The propensity is the project's `fit_propensity` (shared with the matched estimate, so the IPW-ATT and matched ATT differ only in weighting-vs-matching, never in the propensity model). `reg_adjust=True` ports `extrema.md`'s outcome-regression step → the **AIPW / doubly-robust** variant (offered as the credibility-max option). Because the algorithm is the published estimator, its validity is ZSB's theorem; the planted-confounder coverage gate and the `ipw.R` numeric oracle are confirmation, not the sole validity evidence.

### Decoupling diagnostics & NO-GO

`matched_att` = `estimate_att(...).estimate`; `ipw_att` = the `gamma=0` IPW-ATT. `agreement_gap = |matched_att - ipw_att|`; `agreement_ok = gap <= matched_se` (one Abadie–Imbens SE; pinned above). `overlap_ok` = the Hájek weights are stable (`OVERLAP_MIN_ESS_FRACTION` + `OVERLAP_MAX_WEIGHT_SHARE`, pinned above). `msm_go = agreement_ok and overlap_ok`. When `msm_go` is False the MSM band and `gamma_star` are still COMPUTED and returned, but flagged NO-GO in `notes` and NOT reported as a robustness claim for the matched ATT (the E-value still stands). **Point agreement is necessary, not sufficient:** `agreement_ok` matches the two ATT *point* estimates; it does NOT prove the whole IPW-ATT sensitivity CURVE (`gamma_star`) transfers to the matched ATT. The band is a **proxy** gate, and `gamma_star` is reported as a property of the IPW-ATT, never as a certificate for the matched ATT.

### Error handling / edge semantics

- Non-binary `Y` into `e_value` → `ValueError`; into `sensitivity_analysis` → MSM only, `outcome_is_binary=False`, E-value fields `None`, `notes` says why.
- Malformed `gammas` (not ascending, `gammas[0] != 0.0`, negative) → `ValueError`.
- Overlap failure / unstable Hájek weights → `overlap_ok=False`, `msm_go=False`, `notes`; never a silent blow-up (`fit_propensity` already clips scores to `[1e-6, 1-1e-6]`).
- Purity: `Z`, `X`, `Y` never mutated; a new frozen result is returned.

### Harness wiring (the two consumers)

Both harnesses already matched and estimated (`validate_xcross_causal.py:101-114`; the parallel block in `validate_xshot_causal.py`):

1. After the existing `estimate_att`/`estimate_atnt`, call `sensitivity_analysis(Z, X_full, Y, target="att", seed=<the harness seed>)` on the SAME inputs the estimate used.
2. Add the `SensitivityResult` fields to the `metrics.json` payload (the `json.dumps` at `validate_xcross_causal.py:247` and `validate_xshot_causal.py:487`), JSON-safe, with the §What-it-does-NOT-measure caveat text alongside.
3. **Declare the new inputs** in each harness's `declare_inputs(...)` (`validate_xcross_causal.py:34`, `validate_xshot_causal.py:60`) — the Γ grid, `n_boot`, `alpha`, `reg_adjust`, and the sensitivity seed — per ADR-056.

**Consequence (expected, non-failing):** declaring new inputs changes the input-contract digest, so the committed `docs/research/xcross_causal/metrics.json` and `docs/research/tf19_entanglement/metrics.json` (the `validate_xshot_causal` memo) read stale. `tests/scripts/test_input_contracts.py::test_committed_artifacts_still_match_their_declared_inputs` **warns** (`StaleArtifactWarning`), does not fail (`:124-128`). Cleared by the owner-run regen (below), not this cycle.

## What this does NOT measure (construct-validity caveat, per `docs/howto/construct-validity.md`)

The E-value and the MSM quantify robustness to an UNMEASURED confounder of a given strength. They do NOT establish causality, do NOT detect a mis-specified propensity or outcome model, and do NOT cover selection or measurement bias. The **MSM is computed on the Hájek-IPW ATT**: its robustness statement is formally about the IPW-ATT and transfers to the matched ATT only insofar as the two agree — so the result reports the agreement gap and fires NO-GO on divergence beyond the band or on overlap failure. Under Option A the MSM is the *published* ZSB estimator (faithful, not a derived construction), so its validity is the theorem, confirmed by the `ipw.R` oracle and the coverage gate. The **E-value is faithful VanderWeele–Ding** (a closed-form transform of a risk ratio); its RR point and CI are computed on the **IPW estimand** with ZSB's valid percentile bootstrap, with the matched RR a reported companion. `gamma_star` (MSM) is a property of the IPW-ATT and is **never** a matched-ATT certificate — point agreement is necessary, not sufficient for the sensitivity curve to transfer. Each `SensitivityResult` surfaces this boundary in `notes`; the harness memos carry the same text.

## Testing & gates (pre-registered BEFORE any number is seen)

Unit tests in `tests/causal/test_sensitivity.py`:

1. **E-value reproduces VanderWeele & Ding's published worked examples** — exact reproduction of ≥ 2 tabulated (RR, E-value) pairs and one CI-limit case (`RR = 3.9 → E ≈ 7.26`; a CI crossing 1 → `E = 1.0`). The IPW RR point + its ZSB-bootstrap CI reproduce on the oracle fixture (gate 2 machinery); the matched-RR companion is reported but is not the inference basis.
2. **`ipw.R` reference oracle (full ATT path, incl. AIPW)** — the port carries an internal **injection seam** (`fitted_prob=` / `y_fitted=` overrides, test-only) in the Python reimplementation so the oracle drives not just `_get_extrema` but `_extrema_md` / `_extrema_os_att` and the **AIPW add-back**, against fixed propensity / outcome-regression inputs — bypassing `fit_propensity ≠ R-glm`. On a committed fixture of pinned `(A, Y, fitted_prob[, y_fitted], gamma)`, to a tight tolerance: (a) `_get_extrema == get.extrema`; (b) `_extrema_os_att` (`reg_adjust=False`) `== extrema.os("att")`; (c) `_extrema_os_att` (`reg_adjust=True`) `== extrema.os("att", reg.adjust=TRUE)` — so a mis-ported add-back, `rev`, or `1-A` cannot ship undetected. This is the faithfulness gate.
3. **Γ = 1 collapse (automatic)** — `msm_lo[0] == msm_hi[0] == ipw_att`, asserted exactly (follows from the low/high factors coinciding at `gamma = 0`).
4. **Planted-confounder coverage** — a synthetic generator plants a confounder of known strength; across **≥ 500 seed-fixed replications**, the MSM interval at the confounder's true Γ covers the unconfounded truth in **≥ 90%** of replications. Floor (0.90) and reps (500) pinned here, pre-numbers; the plan may tighten, never loosen.
5. **Planted-confounder fixture precondition** (its own test, ADR-056 idiom) — the planted confounder is in-domain, non-degenerate (treatment prevalence off the boundary; it actually moves the naïve estimate), so gate 4 scores signal.
6. **Agreement + overlap NO-GO** — a fixture where matched and IPW ATT agree → `msm_go=True`; a fixture with poor overlap (near-0/1 propensities) → `overlap_ok=False`, `msm_go=False`; a fixture where they diverge beyond the band → `agreement_ok=False`.
7. **Determinism** — same `seed` → byte-identical `SensitivityResult`; a different seed changes the CI, not the point estimates.
8. **Binary guard** — `e_value` raises on continuous `Y`; `sensitivity_analysis` degrades to MSM-only.
9. **AIPW variant** — `reg_adjust=True` runs the outcome-regression path and returns a valid `SensitivityResult` (and reduces to the Hájek path when the OR model is constant).
10. **Purity** — `Z`/`X`/`Y` unmutated; result deterministic.
11. **Harness wiring** — a synthetic `validate_xcross_causal` / `validate_xshot_causal` run emits the sensitivity fields into `metrics.json`, JSON-safe (no corpus; the existing synthetic-fixture path).

Surface gates: `tests/test_public_api_examples.py` (new module + doctest); `tests/scripts/test_input_contracts.py` stays green (warns, not fails); ruff + pyright at CI scope.

## Scope boundary & the owed follow-up

**In this cycle:** `causal/sensitivity.py` + exports + registration; the two harness edits (call + emit + `declare_inputs`); `tests/causal/test_sensitivity.py` + the committed `ipw.R`-oracle fixture; CHANGELOG; this spec + the plan (travel with the one commit).

**Owed, surfaced not dropped (a new TODO row, owner-approved at commit-prep):** an owner-run corpus regeneration of `docs/research/xcross_causal/metrics.json` and `docs/research/tf19_entanglement/metrics.json`, at a clean `main`, to populate the real sensitivity numbers and clear the `StaleArtifactWarning`. CPU, event-level — the StatsBomb/owner corpus the harnesses already consume, **not** the DGX. Deferred to keep THIS cycle a pure code-plus-tests change (a corpus-scale artifact regen is its own step), **not** because it is DGX-blocked; not a quality cut.

**SENS-SPEC-04 — RESOLVED by reading (no fix; final-review Phase 2.5, 2026-10-11):** the r3/B finding claimed `AGENTS.md:126`'s `(ADR-005)` cite for "published-methodology → `NOTICE` entry" was a stray doc-rot. Reading ADR-005 ("Tracking-aware features", PR-S20) overturns that: it DOES carry the attribution practice — it introduces the first tracking-aware features' `NOTICE` entries, lists their academic citations, and says "See `NOTICE` for full bibliographic citations" (lines 217/247). ADR-005 is the ORIGIN of the published-methodology→`NOTICE` discipline (later generalised, de-facto, and recorded in `docs/context/conventions-core.md`, the bullet's other ref). The cite is therefore defensible, NOT the ADR-032→056 class of mis-cite — **no change to that `(ADR-005)` cite**. (The causal bullet does separately gain `sensitivity_analysis` for the new public surface — final-review #2, budget-neutral. Reviewer A r1 blessed the ADR-005 cite correctly; the r3/B flag was raised before reading the ADR body, and A's r2 read confirmed the cite is defensible.)

## Workflow & release

Feature branch `feat/tf71-causal-sensitivity` (no worktree). **One** fully-tested, approval-gated commit (code + tests + spec + plan + CHANGELOG + the oracle fixture together — docs are not a separate or earlier commit). The changed MSM/inference/gates/API sections get a **targeted re-review**; then owner approval; then `writing-plans`; then `/review-plan` (×2); then build; then `/review-impl` (×2); then — **only on an explicit per-commit owner yes** — commit, push, PR, merge after CI green. Additive new public surface → a minor version bump, the number confirmed with the owner at commit-prep (not pre-claimed). No Hyrum break. No retrain, no re-materialize, C4 unchanged (a pure analysis fn). A `NOTICE` entry for VanderWeele & Ding and Zhao–Small–Bhattacharya (a reimplementation of the published ZSB algorithm, with `bootsens/R/ipw.R` acknowledged as the reference-implementation oracle), per the repo's academic-attribution convention (AGENTS.md Key conventions, whose `(ADR-005)` origin cite is defensible — see §Scope boundary SENS-SPEC-04).

## Decisions ruled this cycle

- **D0 (owner, 2026-10-03):** both analyses — E-value AND marginal sensitivity model.
- **D1 (owner, 2026-10-11):** sibling fns + a frozen `SensitivityResult` in `causal/sensitivity.py`; `CausalEstimate` untouched.
- **D2 → REVISED by D5:** the MSM reports a full percentile-bootstrap CI and `gamma_star`.
- **D3 (owner, 2026-10-11):** one inference family (percentile bootstrap), no delta-method path — refined by D5/D6: the MSM and the E-value's RR-CI both use ZSB's valid percentile bootstrap on the IPW estimand (resample + refit, no rematch).
- **D4 (owner, 2026-10-11):** this cycle ships the library seam AND the harness wiring (code, unit-tested on synthetic data); committed memos NOT regenerated — a tracked owner-run follow-up.
- **D5 (owner, 2026-10-11 — the gold-standard ruling):** the MSM is computed on a **Hájek-IPW ATT** by reimplementing the published ZSB algorithm (ZSB's proven estimator + percentile bootstrap), oracle-validated against `bootsens/R/ipw.R`, NOT a derived matching-MSM. The matched ATT stays the headline; the IPW-ATT carries the MSM; the agreement gap + overlap drive a NO-GO. `ipw.R` is a committed reference oracle. The AIPW (`reg_adjust`) variant is offered. Otsu & Rai is moot (it is a wild bootstrap for the matched-ATT *variance*, not an MSM vehicle — the TF-71 row's "weighted/multiplier bootstrap" wording is corrected to "wild bootstrap").
- **D6 (owner, 2026-10-11 — r3 SENS-SPEC-05):** the E-value's RR point AND its CI are computed on the **Hájek-IPW estimand** (valid ZSB percentile bootstrap), NOT on the matched set (whose resample-and-rematch bootstrap is A&I-2008-invalid); the matched RR is reported as a companion only. The NO-GO bars are pinned in-spec pre-numbers — agreement = one Abadie–Imbens SE; overlap = `ESS >= 0.10·n_focal` AND max normalized weight `<= 0.10` (SENS-SPEC-05-B). The `ipw.R` oracle is extended via a test-only `fitted_prob`/`y_fitted` injection seam to verify `_extrema_os_att` + the AIPW add-back, not just `_get_extrema` (SENS-SPEC-06/07).

## References

- VanderWeele TJ, Ding P (2017). *Sensitivity analysis in observational research: introducing the E-value.* Ann Intern Med 167(4):268–274.
- Zhao Q, Small DS, Bhattacharya BB (2019). *Sensitivity analysis for inverse probability weighting estimators via the percentile bootstrap.* JRSS-B 81(4):735–761. arXiv:1711.11286; DOI 10.1111/rssb.12327. Reference implementation: `bootsens` (`R/ipw.R`), github.com/qingyuanzhao/bootsens — the pinned source for the exact tilt + bootstrap.
- Otsu T, Rai Y (2017). *Bootstrap inference of matching estimators for average treatment effects.* JASA 112(520):1720–1732 (a **wild bootstrap** for the matched-ATT variance; **not used** — Option A needs no matching bootstrap).
- Abadie A, Imbens GW (2008). *On the failure of the bootstrap for matching estimators.* Econometrica 76(6):1537–1557 (why a resample-and-rematch bootstrap is invalid for a matching estimator — avoided entirely by putting the MSM on IPW).
- Imbens GW, Rubin DB (2015). *Causal Inference for Statistics…*, Ch. 19 — the Abadie–Imbens matching SE already shipped.
