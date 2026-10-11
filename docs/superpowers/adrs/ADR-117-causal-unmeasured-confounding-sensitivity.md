# ADR-117: Unmeasured-confounding sensitivity on a Hájek-IPW ATT (reimplemented ZSB, oracle-validated against `bootsens`)

| Field | Value |
|---|---|
| **Date** | 2026-10-11 |
| **Status** | Accepted |
| **Deciders** | Karsten |

## Context

`silly_kicks.causal` (ADR-015, promoted as TF-19's second consumer in ADR-037) reports a propensity-matched ATT/ATNT, an Abadie–Imbens SE, SMD balance, and a permuted-block placebo band. It says **nothing about how strong an unmeasured confounder would have to be to explain an estimate away** — the robustness rung of the validity ladder. TF-71 adds that, in the two forms the owner pre-ruled on 2026-10-03: the E-value (VanderWeele & Ding 2017) and the marginal sensitivity model (Zhao–Small–Bhattacharya 2019, "ZSB").

The forcing constraint is the shipped estimator's form: 1:1 nearest-neighbour matching on the logistic propensity, **with replacement, no caliper** (`matching.py:67-68`). Rosenbaum's Γ sensitivity assumes **disjoint matched pairs** (matching *without* replacement), so it does not apply cleanly. ZSB's MSM is proven **for the Hájek-IPW estimator**, not for a matching estimator. And the nonparametric resample-and-rematch bootstrap is **invalid** for matching estimators (Abadie & Imbens 2008) because it fails to reproduce the match-use-count (`K_j`) distribution.

## Decision

Compute **both** the E-value and the ZSB marginal sensitivity model on a **Hájek-IPW ATT** built from the same fitted propensity the matcher uses — a Python **reimplementation of the published ZSB algorithm** (the paper's `get.extrema`/`extrema.md`/`extrema.os`/`bootsens.os` recipe), **oracle-validated against** ZSB's reference implementation `bootsens/R/ipw.R`, with ZSB's own valid percentile bootstrap (resample + refit, no rematch) and an optional AIPW (`reg_adjust`) variant. The **matched ATT stays the headline estimate**; the IPW-ATT carries the sensitivity, and a **matched-vs-IPW agreement gap (one Abadie–Imbens SE) + a Hájek-overlap guard drive a NO-GO** (the MSM is reported, not shipped as a robustness claim, when they diverge or overlap fails). `gamma_star` is a property of the IPW-ATT, never a certificate for the matched ATT. Faithfulness is pinned by a **dev-only, pinned-Docker R run** (`tools/oracle/ipw/`) that regenerates the committed numeric oracle (`tests/causal/fixtures/ipw_oracle.json`); `bootsens` (GPL-3) is **installed into that ephemeral image only, never vendored** into this MIT repo.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Derived matching-MSM (tilt the `K_j` match-count weights) | sensitivity "on the shipped estimator" | ZSB's object is IPW, not a matched estimator; the `K_j` tilt is an unproven worst-case; needs a matching-valid bootstrap (Abadie–Imbens-invalid) | two off-label derivations dressed as fidelity; its validity would rest entirely on a coverage gate, no theorem |
| B. Faithful ZSB IPW-MSM, but **vendor** `ipw.R` into the repo | no network at oracle-regen | `bootsens` is **GPL-3**; vendoring its source into the MIT repo contaminates the license | GPL-3 in an MIT repo; the das-reference precedent pins the external oracle, it does not vendor source |
| C. **Faithful ZSB IPW-MSM; install `bootsens` GPL-3 in the dev-only Docker image; commit only the generated fixture** (chosen) | inherits ZSB's theorem + valid bootstrap; both E-value and MSM faithful + cited; no R/new runtime dep; MIT-clean; reproducible pinned oracle | the MSM is on the IPW-ATT, so it transfers to the matched ATT only insofar as they agree (handled by the NO-GO); a dev-only Docker is needed to regenerate the oracle | — |

Otsu & Rai (2017) is a **wild bootstrap for the matched-ATT sampling variance**, not an MSM vehicle; it is moot under Option C (no matching bootstrap is used).

## Consequences

### Positive

- Both sensitivity analyses are **faithful and cited** (E-value = closed form; MSM = a reimplementation of the published ZSB algorithm, oracle-validated against `bootsens`, on the estimand ZSB proves), rather than a derived construction whose only evidence is a coverage gate.
- **No new runtime dependency** and no R at test time — the native-reimplementation posture of ADR-107/108 (which dropped `accessible-space`), applied to `bootsens`. The repo stays **MIT-clean** (GPL-3 lives only in the ephemeral dev image).
- The faithfulness gate is **non-circular**: the oracle targets come from an independent R run, never the Python port. A pinned `rocker/r-ver` image makes regeneration repeatable (matches the lakehouse Spark-local Docker + the `das-reference` frozen-oracle discipline).

### Negative

- The MSM's robustness statement is formally about the **IPW-ATT**; it transfers to the matched ATT only insofar as the two agree. Mitigated: the result reports the agreement gap and fires a **NO-GO** on divergence beyond one Abadie–Imbens SE or on Hájek-overlap failure, and `gamma_star` is labelled an IPW-ATT property, never a matched-ATT certificate.
- Regenerating the oracle fixture requires **Docker** (first Docker in silly-kicks; a new dev-only `tools/oracle/ipw/` tree). Dev-only — never a runtime/CI-for-tests dependency.
- Wiring the two causal harnesses (`validate_xcross_causal.py`, `validate_xshot_causal.py`) to emit the sensitivity block and declare its inputs (ADR-056) makes the committed `docs/research/xcross_causal/metrics.json` and `docs/research/tf19_entanglement/metrics.json` read stale (`StaleArtifactWarning`, non-failing) until an **owner-run corpus regen** (a tracked TODO follow-up).
- **Derivative-work posture — owner-ruled MIT-clean (2026-10-11).** `sensitivity.py` is an independent reimplementation of the *published* ZSB algorithm (paper + equations); `bootsens` is used only as an out-of-process numeric oracle (GPL-3, installed in the ephemeral dev image, never vendored, never imported at build/test/runtime). The owner ruled this is **not** a GPL-3 derivative-work concern — a reimplementation of a published algorithm is MIT-clean, the same posture already established for the native-DAS reimplementation that dropped `accessible-space` (ADR-107/108). The code and tests carry no `bootsens` source and no runtime GPL-3 dependency.

### Neutral

- `SensitivityResult` is a new frozen public dataclass in `causal/`; `CausalEstimate` is untouched.
- A pinned-Docker R toolchain enters the repo as dev tooling (not the wheel, not CI-for-tests).

## Related

- **Specs:** `docs/superpowers/specs/2026-10-11-tf71-causal-sensitivity-design.md`
- **Plans:** `docs/superpowers/plans/2026-10-11-tf71-causal-sensitivity.md`
- **ADRs:** builds on ADR-015 (causal-validation port) + ADR-037 (TF-19 second consumer); ADR-056 (declared input contracts); the native-reimplementation precedent is ADR-107/108 (native DAS, dropped `accessible-space`); reported-not-gated research is ADR-009.
- **External references:** VanderWeele & Ding (2017), Ann Intern Med 167(4):268–274; Zhao, Small & Bhattacharya (2019), JRSS-B 81(4):735–761 (arXiv:1711.11286; reference impl `bootsens`, GPL-3, github.com/qingyuanzhao/bootsens, pinned commit `6008a8d8`); Otsu & Rai (2017), JASA 112(520); Abadie & Imbens (2008), Econometrica 76(6).
