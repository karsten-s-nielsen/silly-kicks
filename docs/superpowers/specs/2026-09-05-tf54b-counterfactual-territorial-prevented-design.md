# TF-54b — Counterfactual Territorial "Threat Prevented" + a reusable Expected-Passing model

| Field | Value |
|---|---|
| **Date** | 2026-09-05 |
| **Status** | v2.1 — spec review R2 **APPROVE** (SPEC-01..08 resolved, SPEC-09 folded); plan review R1 addressed (worked example §5.7 + cf windowing §5.5 added) |
| **Feature** | TF-54b — the reserved `method="counterfactual"` valuation for `silly_kicks.territory`, plus a new reusable `silly_kicks.expected_passing` component it depends on |
| **Predecessor** | TF-54 v1 (`method="completed_failed"`, shipped 4.108.0 / PR-S179 / ADR-086) |
| **Sizing** | Monstah — own brainstorm → spec → review → plan → review → execute. **One combined cycle** (owner ruling 2026-09-05): the expected-passing model and the territory counterfactual ship together, both validated in-cycle. |

---

## 1. Context & goals

TF-54 v1 (4.108.0) shipped `silly_kicks.territory.compute_territorial_dominance` with a pluggable
`method=` family and a **reserved typed door** `method="counterfactual"` that today raises
`NotImplementedError`. TF-54b implements that door.

**The v1 default and its documented defect.** v1 values every opponent pass whose recorded SPADL `end`
lands inside the defender's trimmed hull at `xT(end)` — conceded if completed, prevented if failed. A
**failed** pass's SPADL `end` is the *death/recovery* location (`_derive_end_coordinates` = next action's
start), **not** the intended target. So v1's "prevented" is a proxy — *"threat that died in the
territory"* — and it **under-counts**: a pass aimed into the zone but intercepted at or before its
boundary has a death location *outside* the hull and is never counted (SHOULD-FIX-1 in the v1 impl
review; the v1 docstring names this as exactly the gap the counterfactual exists to close).

**Goal.** Model the *counterfactual* threat a defender's territory prevented — what an opponent pass into
that territory **would have been worth** had the defender's positioning not blunted it — as a proper
**expected-minus-realized** quantity (the GSAA idiom the house shipped in TF-59, `goals_prevented =
ΣPSxG − GA`). For every opponent pass **aimed into** the territory:

```
expected_threat(pass) = P_complete(pass) · xT(target)      # completion-weighted value at stake
realized_threat(pass) = xT(target) if completed else 0
prevented_above_expectation(pass) = expected − realized = (P_complete − outcome) · xT(target)
```

**Why a completion weight, and why it needs a real model (the spec-review R1 resolution).** `P_complete`
is the "expected" term — without it "expected" assumes every aimed pass completes, over-crediting. The
review (SPEC-01/-06) proved that sourcing `P_complete` from the injected xT's transition matrix `T` is
wrong: `singh_counts` `T` gives only a tiny per-*zone* marginal (≈0.01–0.1 → the headline collapses to
`−conceded`, SPEC-06), and `kde_smoothed` `T` is row-stochastic so it carries **no** completion signal
at all (SPEC-01). A proper per-pass `P_complete` (≈0.2–0.9) needs a **fitted expected-passing model**,
and for a **failed** pass it must be evaluated at the *hypothesized* target (its distance/angle are the
*intended* geometry, not the truncated death geometry), integrated over the target uncertainty:

```
expected_threat(failed pass) = Σ_z  q(z | origin, death-cone) · c(origin, z) · xT(z)
```

- `q(z | origin, death-cone)` — the **target distribution** from the injected xT's transition `T`,
  restricted to the death-direction cone and renormalized (the owner-approved Tier-2 reuse — `T` is used
  **only** as a destination *distribution*, which both xT families provide validly, so **SPEC-01
  dissolves**: no reliance on `T`'s family-specific row-mass, no fail-closed guard).
- `c(origin, z)` — a proper **pass-completion probability** from the new `PassCompletionModel`,
  evaluated at each hypothesized target `z` (**SPEC-06 dissolves**: a real per-pass `c`, non-degenerate).
- `xT(z)` — value at the target from the injected xT (`values_at_points`).

For a **completed** pass the target is observed, so no integration: `expected = c(origin, end)·xT(end)`,
`realized = xT(end)`, contribution `(c(origin,end) − 1)·xT(end)` — the defender is blamed for conceding
passes that *usually fail* (low `c`) and absolved for the unstoppable ones (high `c`), the correct
defensive reading.

**This is one combined cycle** delivering a **reusable `PassCompletionModel`** (event-only expected
passing — useful well beyond territory) plus the territory counterfactual that consumes it.

**The load-bearing constraint (why this is not a second xt-gk-v2).** An event-only counterfactual is the
GKDV/ghosting problem in a new setting; xt-gk-v2 shipped a possession-value surface with **no ground
truth** and was quarantined (`docs/research/xtgk_v2_construct_validity/`). TF-54b is different because
**every component has true ground truth**: `c` validates on real completed/failed outcomes (held-out
AUC/calibration/Brier), `q`'s target recovery validates via a synthetic-interception substrate (§7), and
the composed metric runs a pre-registered battery. The method ships **behind the non-default door with
the report attached**; `completed_failed` **stays the default**; promotion is a *separate* ADR-009 apply
decision after the owner reads the report (reported-not-gated). "Gold standard" = principled **and**
validated.

## 2. Non-goals / scope boundaries

- **Not** a change to `completed_failed` — byte-identical (same output shape and values under the default
  method). No VAEP/tracking retrain, no re-materialize.
- **Not** a promotion of the default. Ships available-but-non-default; flipping the default is a
  follow-on ADR-009 decision gated on the owner reading the validation report.
- **Not** a possession-value ghosting surface (xt-gk-v2 `V(z,p)`). Reserved as a *further* target-valuator
  upgrade (`V(z)` for `xT`) or an additional `method=` value — never this headline (its construct is the
  quarantined one; §6).
- **Not** a tracking metric. Event-only throughout; `territory` and `expected_passing` import `spadl` /
  `id_compat` / (`xthreat` for territory) only, **never** `tracking` (AST import-allowlist gates).
- **Not** a deterrence metric. A dominant territory means opponents *don't attempt* passes into it;
  un-attempted passes are **unobservable event-only**. TF-54b measures attempted-and-blunted passes only;
  the deterrence ceiling is a **documented limitation** in docstring/glossary/report.

## 3. Global constraints

- **Event-only import graph** — enforced by AST allowlists (`tests/territory/test_import_allowlist.py`
  extended for the completion-model port; a new `tests/expected_passing/test_import_allowlist.py`).
- **Injected fitted ports** — both `xt: ExpectedThreat` and the completion model are injected
  (`TYPE_CHECKING`-only import + duck-typed, ADR-022); `require_fitted_xt` and a completion-model
  fail-closed load guard reject unfitted/None/str. Value lookups use the **public** `values_at_points` /
  the new xthreat seam (§5c), **never** raw `.xT`/`.transition_matrix` indexing, **never** `rate`.
- **Bundled-trained-artifact discipline (ADR-011/016/040/044/050)** for `PassCompletionModel`: pickle-free
  JSON + `SHA256SUMS`, feature contract, chirality probe, fail-closed load, inference imports no sklearn.
- **Canonical-id grouping** (ADR-019), **drop-and-count** (ADR-042), **ADR-028** reflection, **purity**
  (ADR-033), **artifact provenance** (`scripts/_provenance.py` `require_clean_tree` + `run_commit` /
  `training_commit`; ADR-052 `for_each` shards; ADR-056 input contract).
- **`for_provider` ships EMPTY** (ADR-009) for `TerritoryParams`, `CounterfactualParams`, and any
  completion-model per-provider hook.

## 4. Architecture overview

```
silly_kicks/expected_passing/            # NEW reusable component (event-only)
  __init__.py            # PassCompletionModel + fit/predict/save/load surface
  _model.py              # logistic (pure-numpy serve) + JSON/SHA256 serialization + load guards
  _features.py           # event-only pass-completion feature extractor + feature contract
  weights/               # bundled default weights (JSON + SHA256SUMS), public-corpus-trained
silly_kicks/xthreat/
  _physical.py / __init__.py             # + public seam: cone-restricted destination distribution (§5c)
silly_kicks/territory/
  _columns.py            # + per-method column/dtype resolver + counterfactual-only columns
  _config.py             # + CounterfactualParams (frozen)
  _report.py             # + counterfactual census fields
  _compute.py            # method dispatch: completed_failed (unchanged) | counterfactual -> _counterfactual
  _counterfactual.py     # NEW — the joint valuation: q (xthreat seam) x c (PassCompletionModel) x xT
scripts/
  train_pass_completion.py               # NEW — owner-run trainer (for_each shards + provenance)
  validate_territory_counterfactual.py   # NEW — owner-run construct-validity corpus pass
docs/research/territory_counterfactual_construct_validity/   # NEW — validation artifact (Commit 2)
```

C4: **a new `expected_passing` container** (dot re-render). The **33 action-coupled `add_*` aggregator
count is unchanged** — neither the counterfactual method nor the completion model is an `add_*`.

## 5. Feature design — `method="counterfactual"` (Option D, joint model)

### 5.1 The quantity

The unified expected − realized counterfactual of §1, summed over opponent passes **aimed into** the
territory. `xt_conceded` = realized threat of completed passes (formula-identical to v1 for the *value*;
membership rule pinned in §5.3). `xt_prevented` (counterfactual) = `Σ_{failed aimed-in} Σ_z q·c·xT`.
Net headline `xt_prevented_above_expectation` = `Σ (P_complete − outcome)·xT(target)`.

### 5.2 The three components

- **Target distribution `q` (Tier-2, injected xT).** For a failed pass from origin cell `i` with observed
  death direction `θ`: `q(z) ∝ T[i, z]` over territory zones `z ∈ R` lying in the cone of half-angle
  `direction_cone_degrees` around `θ`, renormalized. Sourced through the **public xthreat seam** (§5c) —
  never raw `T`. Used only as a distribution (family-agnostic). **Death-direction is a noisy proxy
  (SPEC-07):** the death location of an *interception* is where the defender redirected the ball, not a
  pure flight terminus, so `θ` is an approximation of the intended direction — which is exactly why `q`
  spreads mass over a *cone* rather than a ray. The synthetic-interception test (§7) perturbs the
  synthetic death in **both** distance and direction (SPEC-09), so it validates `direction_cone_degrees`
  against known ground truth rather than only distance recovery; the composed real-data leg additionally
  carries genuine interception noise. Named in the docstring + report, not silent.
- **Completion `c` (new `PassCompletionModel`, §5b).** `c(origin, z) = P(complete | pass geometry
  origin→z)`, evaluated at each hypothesized target `z` (failed) or the observed end (completed).
- **Value `xT(z)` (injected xT).** `values_at_points` / the seam at each `z`.

**Composition.** Failed pass: `expected = Σ_{z∈R∩cone} q(z)·c(origin,z)·xT(z)`. Completed pass:
`expected = c(origin,end)·xT(end)`, `realized = xT(end)`. A pass whose cone∩R has no transition support
(`Σ T = 0`) or whose completion features are non-finite → **unresolvable target**, dropped-and-counted
(`territory_target_source="unresolved"`), never a fabricated 0.

### 5.3 Membership, columns & schema (SPEC-04 pinned)

**Membership rules (pinned so conceded stays comparable to v1):**
- **Completed pass:** counts iff its observed `end` is **point-in-hull** (reflected into the defender
  frame, the v1 rule) — so `xt_conceded`'s *value* is formula-identical to v1 (same passes, same
  `xT(end)`).
- **Failed pass:** counts iff its death-direction cone from the origin **intersects** the reflected hull
  region `R` (the re-opened membership — includes aimed-in-died-short). The zone set `R` = xT grid cells
  whose centres fall in the reflected hull.

**Method-dependent schema** (counterfactual-only columns appear only under `counterfactual`, so
`completed_failed`'s shape/values are untouched):

| Column | `completed_failed` | `counterfactual` |
|---|---|---|
| `territory_xt_conceded` | v1 (realized, completed-in-hull) | same value (realized, completed-in-hull) |
| `territory_xt_prevented` | v1 (`Σ xT(death)`, failed-in-hull) | `Σ_{failed aimed-in} Σ_z q·c·xT` |
| `territory_xt_net` | v1 (`conceded − prevented`) | `conceded − prevented` (counterfactual prevented) |
| `_xt_conceded_forward` / `_prevented_forward` | v1 | counterfactual analogs |
| `territory_passes_into_hull` | v1 (observed end in hull) | **same v1 meaning** (descriptive, method-invariant) |
| `_xt_conceded_rate` / `_xt_prevented_rate` | v1 (`/passes_into_hull`) | **`/passes_aimed_into_hull`** (documented method-dependent denominator) |
| geometry (`hull_area`, `centroid_*`, `def_actions_in_hull`), `hull_source` | v1 | v1 |
| `territory_expected_threat_faced` (cf-only) | — | `Σ P_complete·xT(target)` over aimed-in passes |
| `territory_xt_prevented_above_expectation` (cf-only) | — | GSAA headline `Σ (P_complete − outcome)·xT(target)` |
| `territory_passes_aimed_into_hull` (cf-only) | — | the counterfactual scoring denominator |
| `territory_mean_completion_faced` (cf-only) | — | mean `c` over aimed-in passes (interpretability) |
| `territory_target_source` (cf-only) | — | `{observed, modeled, unresolved}` provenance |

Dtypes: ids/provenance `object`; counts `Int64`; xT/area/coords/rates/probabilities `float64`. A
`_columns.py` resolver returns the column/dtype dict **by method**; schema/liveness/glossary gates iterate
the per-method set. Each column's per-method definition is glossaried (with `higher_is_better`) so no two
columns silently mean the same thing (§10 pin extended to the passes-count pair and the rate denominators).

### 5.4 `CounterfactualParams`

Frozen, alongside `TerritoryParams`, resolved by `method` (`.default`/`.for_provider`/`.is_default`
idiom). Knobs (universal-safe defaults, no tuning this cycle per ADR-009; a distinct value is a reserved
knob): `direction_cone_degrees`, `min_transition_support` (below which target is `unresolved`),
`target_zone_grid` (defaults to `xt.grid`). `_PROVIDER_COUNTERFACTUAL_PARAMS` ships **empty**.

### 5.5 Windowing

Additive pooling over a `window` of `game_id`s, hull re-derived over pooled actions (v1 semantics). The
counterfactual columns pool by their **kind**, pinned so nothing is a mean-of-means:
- **Sums** (pool by adding): `xt_conceded`, `xt_prevented`, `xt_net`, `_forward` companions,
  `expected_threat_faced`, `xt_prevented_above_expectation`, `passes_into_hull`, `passes_aimed_into_hull`.
- **Rates** (re-derive from pooled sums, never average per-match rates): `xt_conceded_rate = Σconceded /
  Σpasses_aimed_into_hull`, `xt_prevented_rate = Σprevented / Σpasses_aimed_into_hull`.
- **`mean_completion_faced`** pools as **`Σ(c over aimed-in passes) / Σ(aimed-in passes)`** across the
  window — a support-weighted mean, NOT the mean of per-match means (a low-volume match must not weigh
  equal to a high-volume one).
- **`target_source`** is per-pass provenance; at window grain it is dropped (or summarized as the modeled
  fraction) — a window row reports the pooled counts, not a single per-row token.

### 5.7 Worked example (the exact golden the core test pins — PLAN-01)

Defender A (attacks x=105 in action-LTR); opponent B's passes are reflected `(105−x, 68−y)` into A's
frame for membership. Take completion `c = 0.6` (constant, for a hand-checkable expectation) and two
opponent passes into A's hull:

- **Completed** pass, reflected end inside the hull, with `xT(end) = 0.15`:
  `conceded += 0.15`; `expected_faced += c·0.15 = 0.09`; above-expectation contribution
  `(c − 1)·0.15 = −0.06`.
- **Failed** pass whose death-direction cone ∩ hull selects two zones with renormalized
  `q = (0.7, 0.3)` and values `xT = (0.10, 0.20)`:
  `prevented += c·Σ q·xT = 0.6·(0.7·0.10 + 0.3·0.20) = 0.6·0.13 = 0.078`;
  `expected_faced += 0.078`; above-expectation contribution `+0.078`.

Totals: `conceded = 0.15`, `prevented = 0.078`, `expected_threat_faced = 0.09 + 0.078 = 0.168`,
`passes_aimed_into_hull = 2`, `mean_completion_faced = 0.6`,
`xt_prevented_above_expectation = expected_faced − conceded = 0.168 − 0.15 = 0.018`
(equivalently `0.078 + (−0.06)`). A **uniform-xT** degenerate check: if every relevant zone has
`xT = 0.1`, the failed-pass `prevented = c·0.1·Σq = 0.6·0.1·1 = 0.06` **exactly** (renormalization to 1 ×
the completion weight) — the minimal invariant the core test asserts to the last digit.

## 5b. The `PassCompletionModel` (new reusable component)

- **Type.** Logistic regression with a pure-numpy `sigmoid(Xβ)` serve (the `GkCompletionModel` idiom,
  in an event-only home), pickle-free JSON + `SHA256SUMS`, feature contract + chirality probe, fail-closed
  load; inference imports no sklearn. (A GBM variant is a reserved door if the logistic is under-powered —
  decided by its own held-out validation, not up front.)
- **Features (event-only).** Origin `(x,y)`, target `(x,y)`, distance, angle-to-goal, forward component,
  lateral component, and origin pitch-third / game-state as available in SPADL — the standard
  expected-passing feature set. No tracking, no teammate positions.
- **Training.** On observed pass geometry, the field standard: completed passes at their real `end`
  (label 1), failed passes at their death location (label 0). Public-corpus-trained bundled default
  (redistributable — public data only). `train_pass_completion.py` is a `for_each`-sharded, provenance-
  stamping owner-run trainer; `training_commit` stamps the code commit.
- **Inference for the counterfactual (stated interpolation).** The model is *evaluated* at hypothesized
  targets `z` for failed passes — points **further downfield** than the death location. This is an
  interpolation **within the geometry range completed passes already cover** (completed passes reach such
  targets), not wild extrapolation; the assumption is stated in the docstring and probed by the model's own
  held-out validation across the geometry range. The failed-pass *aim* prior (that intended targets
  resemble where successful moves went, via `q`) is the one irreducible event-only assumption, stated and
  probed by the synthetic-interception test (§7).

## 5c. The public xthreat seam (SPEC-03)

Instead of `territory` reaching into `xt.transition_matrix` (raw, flat-indexed, y-inverted,
family-specific — the ADR-041 defect class one module over), expose a **public** `xthreat` helper that
returns, for an origin point and a zone set, the **renormalized destination distribution** over those
zones together with their physical centres and `xT` values — with the flat-index + y-inversion handled
**inside** `xthreat` (where the convention is owned) and family-agnostic (a distribution, not a row-mass).
`territory` consumes only this seam. Mirrors `values_at_points`/`physical_grid`. The wrong
`singh_transition_matrix` "row-stochastic" docstring is corrected upstream in the same change (SPEC-05),
and the seam's own tests pin the flat-index/y-inversion contract.

## 6. Rejected alternatives

| Option | Why rejected |
|---|---|
| **Drop the completion weight (value only the intended target)** | A de-scope: "expected" then assumes every aimed pass completes, over-crediting. The completion weight is core to a proper expected-threat; do it right (owner ruling, 2026-09-05). |
| **Source `P_complete` from the injected xT's transition `T`** | SPEC-06: `singh` per-zone marginal ≈0.01–0.1 → headline collapses to `−conceded`. SPEC-01: `kde` `T` is row-stochastic → no completion signal → silently wrong. A per-pass `P_complete` needs a fitted model. |
| **Fail-closed on a non-`singh` xT** | Unnecessary once completion is a separate model: `T` is used only as a *distribution*, valid for both families. Fail-closed would artificially restrict which calibrated xT a user may inject. |
| **Evaluate `c` at the death geometry / origin-only** | Death geometry is truncated (biases `c` low on exactly the failed passes); origin-only can't distinguish a short vs long intended pass. Evaluate `c` at the *hypothesized* target, integrated over `q`. |
| **Possession-value ghosting surface (`V(z,p)`)** | Its construct failed out-of-sample and was quarantined; duplicates xt-gk-v2/VAEP. Reserved as a target-valuator swap. |
| **Split into PR-1 (completion model) + PR-2 (territory)** | The owner chose one combined cycle; both ship, co-designed and co-validated. |
| **Same columns, method changes values, no new columns** | Loses the "expected faced" / "above expectation" names; method-dependent schema keeps them explicit while preserving byte-identity. |
| **Promote counterfactual to default in-cycle if it validates** | Reported-not-gated (ADR-009): a separate apply decision after the owner reads the evidence. |

## 7. Validation design (pre-registered, in-cycle, reported-not-gated)

Every component has ground truth, so each is validated at its own level, then the composed metric.

### 7.1 Substrate & drivers

- Public leg: **StatsBomb WC2022 open data** (v1 e2e substrate; contains the elite defenders). Optional
  owner-tier cross-check: **Gradient Sports WC2022 via pining**.
- `train_pass_completion.py` and `validate_territory_counterfactual.py` — `for_each` shards (ADR-052),
  `require_clean_tree` + provenance stamp, input contract (ADR-056). xT and the completion model are fit
  on a corpus **disjoint** from the scored matches (leakage-guarded); the validation fits its own
  completion model (does not depend on the bundled weights) so provenance is clean.

### 7.2 Pre-registered test battery

**Component-level (true ground truth):**
- **`PassCompletionModel`** — held-out **AUC / calibration (ECE) / Brier** on real completed/failed
  outcomes, across the geometry range (so the failed-pass hypothesized-target interpolation is probed, not
  assumed). Pre-registered floors (mirroring `GkRetentionModel`'s `ece ≤ 0.10`-style gate).
- **Target recovery (`q`) — synthetic-interception substrate (SPEC-02 fix, SPEC-09 strengthened).** Take a
  *completed* pass (true end = ground truth), synthesize an interception at a random flight-fraction
  `f∈(0,1)` along origin→end **and a random angular offset `δ` off the ray** (so both the intended
  *distance* **and** *direction* are corrupted — the angular perturbation is what exercises and validates
  `direction_cone_degrees` against known ground truth), take that as the "death" the estimator sees, hide
  the true end, and require the direction-conditioned estimator to recover it **better than baseline (a)
  "death = the synthesized intercept"** (now *not* exact) and **(b) origin-zone centroid**. Leakage-free
  (death-direction from the synthetic perturbed intercept, not the hidden end) and discriminating. The
  composed Primary-1 additionally runs the held-out **real-data** leg, whose interceptions carry genuine
  (not synthetic) direction noise.

**Composed-metric primary — decides promotion (both must clear):**
1. **Mechanism validity** — the composed counterfactual, driven by the validated `c` and `q`, beats the
   v1 death-location proxy and the naive baselines on the synthetic-interception + held-out substrate.
2. **Face validity — the "Van Dijk" prior.** A **locked, pre-registered elite-defender list** (committed
   as a constant in Commit 1, the TF-19 `NAMED_KEEPER_PRIOR` idiom, stamped in the artifact) surfaces in
   the top decile/quartile at meaningful pass-faced volume.

**Composed-metric secondary — reported, not gating:**
3. **Reliability** — split-half / across-window stability.
4. **Discriminant** — moves meaningfully vs v1 `completed_failed`, vs volume (`passes_aimed_into_hull`),
   vs team defensive strength; beats a shuffled-outcome placebo. **Includes a decomposition-non-degeneracy
   check (SPEC-06):** confirm the concede/prevent split is not degenerate under the chosen completion model
   (i.e. the metric is not ≈`−conceded`).
5. **Outcome-lens** — possession-reaches-shot AUC reported for continuity, **explicitly not gating** (a
   metric of threat *denied* is not scored on threat *produced* — the lens that unfairly sank xt-gk-v2).

**Decision rule.** Promote iff composed-Primary-1 **and** -2 clear with Secondary consistent, *and* the
`PassCompletionModel` cleared its own floors; else the method ships available-but-experimental and the
report says so. Promotion is a follow-on ADR-009 apply decision.

## 8. Testing (two tiers)

**Regular suite (`-m "not e2e"`, committed fixtures):**
- `PassCompletionModel`: fit/predict on a toy fixture; pure-numpy serve == training-lib prediction;
  JSON+SHA256 round-trip; **load guards** (chirality mismatch raises, missing/altered feature contract
  raises per ADR-050/044); a hand-computed logistic value; NaN-feature → NaN/`unresolved` (never a
  fabricated prob). Import-allowlist (event-only) with planted-violation meta-tests.
- The public xthreat seam: hand-computed destination distribution on a toy grid; **flat-index + y-inversion
  contract pinned**; family-agnostic (singh and kde both give a valid distribution); SPEC-05 docstring fix.
- `_counterfactual.py`: the joint `Σ q·c·xT` against hand-computed values on toy `xt` + toy `c`;
  **`completed_failed` byte-identity** (the load-bearing additive guard); method dispatch + method-dependent
  schema; membership rules (completed point-in-hull; failed cone∩R; aimed-in-died-short counted;
  unresolvable-target dropped-and-counted with report conservation).
- Reflection-invariance (per-row; one scene from either perspective scores identically), incl. a failed
  pass whose death ≠ intended target.
- **Mechanism-recovery-beats-naive on a fixture** (the synthetic-interception offline half — `q` beats
  "death=intercept" and "centroid").
- Purity (ADR-033, ≥2 variants); territory import-allowlist extended for the completion-model port;
  glossary coverage for new columns; NOTICE citations.

**e2e (`@e2e`, owner-run):** the full corpus training + validation (§7) producing the artifact.

## 9. Commit plan

**One feature branch off `main`. Single commit unless provenance needs the second.**

- **Commit 1 — all code + release.** `expected_passing/` (`PassCompletionModel` + features + contract),
  the xthreat seam + SPEC-05 docstring fix, `territory/_counterfactual.py` + dispatch + method-dependent
  `_columns.py` + `CounterfactualParams` + `_report.py`, the **locked elite-defender prior constant**,
  both `scripts/` drivers, all offline tests (§8 — using fixture-fit models, not bundled weights),
  `feature_glossary` + `glossary_emitted_columns` + `NOTICE`, C4 `dot` re-render (new `expected_passing`
  container; 33 aggregators unchanged), version bump, `CHANGELOG`, the new ADR. Fully green: `-m "not e2e"`
  + ruff + ruff format + bare pyright. **Default stays `completed_failed`.**
- **Owner-run at Commit 1 (clean tree, `--out` outside the repo):** (a) train `PassCompletionModel` on the
  public corpus → bundled weights (`training_commit` = Commit 1); (b) run the territory-counterfactual
  validation corpus pass (fits its own leakage-disjoint models) → `metrics.json` + `named_defender_signs.
  parquet` + `findings.md` (`run_commit` = Commit 1).
- **Commit 2 — the two provenance artifacts** (the bundled `expected_passing/weights/` JSON+SHA256 **and**
  `docs/research/territory_counterfactual_construct_validity/`), **both stamping Commit 1**. This is the
  single justified second commit: committing either with the code would force a dirty-tree stamp, which the
  provenance discipline forbids. Load-bearing non-squash so the stamps resolve.

Then owner-driven, explicit approval at every gate (never a commit/push/tag without an unambiguous yes):
push → PR → CI green → admin-merge non-squash → tag → PyPI.

**Numbers re-derived at commit-prep** (`git fetch && git merge origin/main`) — TF-60 PR5 ghost-outfield
(`feat/tf60-pr5-ghost-outfield`) is unmerged; if it lands first this renumbers. Provisional: 4.109.0 /
PR-S180 / ADR-087.

## 10. Open / plan-time details

- `PassCompletionModel` feature list finalization (which game-state features SPADL exposes event-only) and
  the logistic-vs-GBM decision (decided by held-out AUC/Brier, not up front).
- The xthreat seam's exact signature (return shape: zones × probabilities × centres × values) and name.
- `direction_cone_degrees` / `min_transition_support` universal-safe defaults + a sensitivity note (ADR-009:
  no tuned default; a reserved knob).
- Completed-pass `c` domain check — confirm the observed-end geometry is in-domain (it is, by construction)
  and the failed-pass hypothesized-target geometry stays within the model's validated range.
- `_report.py` counterfactual census fields (`n_target_modeled`, `n_target_unresolved`, …) + conservation.
- Bundled-weights redistribution check (public-corpus-trained ⇒ in-repo JSON; confirm no licensed data).
- Elite-defender prior list contents (locked before the run) + the top-quantile threshold.
- `xt_net` vs `xt_prevented_above_expectation` sign convention + `higher_is_better` per column (SPEC-04) —
  pin each so no two columns silently coincide.

## 11. References (NOTICE)

- Singh, Karun. "Introducing Expected Threat (xT)." 2019 — the transition/value-iteration engine.
- Sumpter, D. *Soccermatics* / Twelve module 10.2 (the "Van Dijk" territorial-dominance lens) — TF-54.
- Expected-passing / pass-completion modelling (the field-standard "expected pass") — `PassCompletionModel`.
- The GSAA "expected − realized" framing — the house's TF-59 shot-stopping (ADR-085) analog.
- Counterfactual defensive valuation (ghosting: Le et al. 2017) — cited as the *reserved* comparator, not
  implemented.

## 12. Decision log

- **2026-09-05 (brainstorm + spec-review R1, owner):** quantity = **Option D**, the completion-weighted
  expected − realized counterfactual (GSAA-analog). Completion is **not** dropped (owner: not a de-scope)
  and **not** sourced from `T` (SPEC-01/-06); it comes from a **new reusable `PassCompletionModel`**
  evaluated at hypothesized targets, integrated over a target distribution `q` taken from the injected xT's
  transition (Tier-2, used only as a family-agnostic distribution). Value from the injected xT. New public
  **xthreat seam** owns the flat-index/y-inversion (SPEC-03) + SPEC-05 docstring fix. Validation is
  **in-cycle owner-run, reported-not-gated**: component-level (completion AUC/ECE/Brier; `q` recovery via
  the **synthetic-interception** substrate, SPEC-02) then composed (mechanism + elite-defender prior
  primary; reliability/discriminant/non-degeneracy/outcome-lens secondary). Packaging = method-dependent
  schema, `completed_failed` byte-identical, columns pinned (SPEC-04). Architecture = new `expected_passing`
  C4 container; 33 aggregators unchanged. Cycle = **one combined cycle** (owner ruling), **Commit 1 code +
  Commit 2 provenance artifacts** (bundled weights + validation), promotion a follow-on ADR-009 decision.
