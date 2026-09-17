# TF-54b Path B (revival) — event-only counterfactual territorial "threat prevented" + broad-corpus validity + defender-ranking census

**Status:** design (brainstorm-approved 2026-09-17; owner cleared scope/breaking, "gold standard").
**Release coordinates:** the version / PR-Snnn are **NOT locked until Commit 2** (owner rule — another
session may take a number first; re-derive from `main` at commit-prep). Any number in this doc is
illustrative only. The **ADR number is NOT locked** — docs use the placeholder `ADR-NNN`; the real number
is assigned at release-prep (after `git fetch && git merge origin/main`), because another session may
release first. (The branch's ADR-089 collides with `main`'s TF-60 Layer-3 ADR-089, and ADR-090 is the
tracking sibling, so whatever number is free must be freshly taken then.)
**Predecessor design (settled baseline, treated as-is):** `2026-09-05-tf54b-counterfactual-territorial-prevented-design.md`
(reached spec-review **R2 APPROVE**; lives on the unmerged branch
`origin/feat/tf54b-counterfactual-territorial-prevented`, commit `ab9001c`).

---

## 1. Context & goals

TF-54 v1 (4.108.0) shipped `silly_kicks.territory.compute_territorial_dominance` with a pluggable
`method=` family and a **reserved typed door** `method="counterfactual"`. In the intervening cycles that
door was **removed** on `main` (ADR-090 gave the "counterfactual" *name* to the tracking-consuming
`silly_kicks.territorial_defense` sibling, and the event-only cone was deliberately **not** carried when
the reusable seams were): today `TERRITORY_METHODS == frozenset({"completed_failed"})` and
`compute_territorial_dominance(actions, xt, method, window, params)` carries **no** `completion_model`
parameter. **This revival re-adds the door and fills it with the real event-only implementation.**

Chesterton's Fence: the door was removed because it was a dead `NotImplementedError` after ADR-090 took
the "counterfactual" name for the tracking path — **not** because the cone was rejected. The event-only
cone is a *distinct mechanism* (no pitch control) and coexists with `territorial_defense` under the
TF-54b family. ADR-NNN records this explicitly so a future reader does not read the re-add as reverting
ADR-090.

**The v1 default and its documented defect.** v1 (`completed_failed`) values every opponent pass whose
recorded SPADL `end` lands inside the defender's trimmed hull at `xT(end)` — conceded if completed,
prevented if failed. A **failed** pass's SPADL `end` is the *death/recovery* location, **not** the
intended target, so v1's "prevented" is a proxy that under-counts (a pass aimed into the zone but
intercepted at/before its boundary dies *outside* the hull and is never counted).

**Goal.** Model the *counterfactual* threat a defender's territory prevented — what an opponent pass into
that territory **would have been worth** had the defender's positioning not blunted it — as an
**expected − realized** quantity (the GSAA idiom the house shipped in TF-59, `goals_prevented = ΣPSxG − GA`).
For every opponent pass **aimed into** the territory:

```
expected_threat(pass) = P_complete(pass) · xT(target)
realized_threat(pass) = xT(target) if completed else 0
prevented_above_expectation(pass) = (P_complete − outcome) · xT(target)
```

**Why the counterfactual is *validatable* where xt-gk-v2 was not (a construction argument, not a result).**
Every component has ground truth: `P_complete` (`c`) validates on real completed/failed outcomes (held-out
AUC/ECE/Brier); the failed-pass target distribution `q` validates on a synthetic-interception substrate;
the composed metric runs a pre-registered battery. It ships behind the non-default door with the report
attached; `completed_failed` stays the default; promotion of the default is a *separate* ADR-009 apply.

**Empirical status of the predecessor — stated honestly (SPEC-01).** The baseline design reached
spec-review **R2 APPROVE**, which is a *design* verdict only. The cone was **never merged and never run** —
there is no `docs/research/territory_counterfactual_construct_validity/` on `main` (only the SB360 *sibling*
`territorial_defense_construct_validity/` exists), so **no empirical prior supports the headline**. Because
`territory_xt_conceded` is formula-identical to v1 (§5.3), the composed **Primary-1 "beats v1" margin lives
ENTIRELY in the failed-pass (prevented) leg** — a **null result is a real, un-gated risk**, and this
cycle's validation must be able to *report a null* (reported-not-gated; the default never flips on a null).
**What broad corpus does and does not buy:** more failed-pass-into-hull volume and geometry diversity make
Primary-1 better **powered** and the `PassCompletionModel` better fit — it does **not** guarantee a
positive margin. This is a **separate axis** from the ranking-feasibility that the broad corpus's
multi-season club structure enables (§8): Primary-1 asks *does the mechanism beat v1's proxy*; the census
asks *can defenders be separated from teams*. Neither implies the other.

**What this revival adds beyond the R2-approved baseline (the deltas):**

1. **Re-add the removed door** (§5).
2. **Broad public corpus** — validate and re-bundle on the full StatsBomb open-data corpus
   (`scripts/_sb_open_data.all_open_competitions()`), **not** WC2022-only (§7). Note
   `all_open_competitions()` returns `(competition_id, season_id)` **tuples**, not matches; the *match*
   count is whatever the open-data manifest holds **at run time** and is re-counted then. TF-53 measured
   **~3,961 matches** at its run (`docs/research/tf53_match_outcome_calibration/metrics.json`
   `n_matches: 3961`; TF-52 the same order) — used here as a size **estimate**, not a fixed property.
   **Match count feeds fitting/scoring power; it does NOT establish ranking identifiability** — that is a
   corpus-*composition* question the census measures (§8).
3. **Re-fit + re-bundle `PassCompletionModel`** on that broad corpus (owner decision 2026-09-17),
   replacing the WC2022-trained bundled weights already on `main` (§5b, §9).
4. **Defender-ranking census + gate** — a new pre-registered crossed defender+team-cell census that
   decides, **on measured evidence**, whether a defender **ranking** is licensed on the broad corpus; the
   ranking is published **in-cycle iff the gate clears** (owner decision 2026-09-17), else the cycle ships
   the metric only with the census reported (§8).
5. **Fresh ADR-NNN** (the branch's ADR-089 collides with `main`'s TF-60 Layer-3 ADR-089; `main`'s ADR-090
   is the tracking sibling).

## 2. Non-goals / scope boundaries

- **Not** a change to `completed_failed` — byte-identical output shape and values under the default method.
  No VAEP/tracking retrain, no re-materialize.
- **Not** a change to `territorial_defense` (ADR-090, the tracking Path A). The two TF-54b siblings coexist.
- **Not** a promotion of the `counterfactual` default. Ships available-but-non-default; flipping the
  default is a follow-on ADR-009 decision after the owner reads the validation report.
- **Not** a possession-value ghosting surface (xt-gk-v2 `V(z,p)`). Reserved as a *further* target-valuator
  swap (`V(z)` for `xT`) or an additional `method=`, never this headline.
- **Not** a tracking metric. Event-only throughout; `territory` and `expected_passing` import `spadl` /
  `id_compat` / (`xthreat` for territory) only, **never** `tracking` (AST import-allowlist gates).
- **Not** a deterrence metric. Un-attempted passes into a dominant territory are unobservable event-only;
  TF-54b measures attempted-and-blunted passes only. The deterrence ceiling is a documented limitation.
- **Not** a new library ranking API. Per ADR-009 rankings stay consumer-side. "Ship the ranking" (§8) means
  publishing a *validated reported ranking artifact* plus the ADR-009 apply that records the metric is
  player-attributable **on this corpus** — the per-`(defender, match)` metric columns do not change and no
  ranking function is added to the library surface.

## 3. Global constraints

- **Event-only import graph** — AST allowlists (`tests/territory/test_import_allowlist.py` extended for the
  completion-model port; `tests/expected_passing/test_import_allowlist.py`; the census driver imports no
  `tracking`).
- **Injected fitted ports** — both `xt: ExpectedThreat` and the completion model are injected
  (`TYPE_CHECKING`-only import + duck-typed, ADR-022); `require_fitted_xt` + a completion-model fail-closed
  load guard reject unfitted/None/str. Value lookups use the **public** `values_at_points` / the xthreat
  destination-profiles seam, never raw `.xT`/`.transition_matrix` indexing, never `rate`.
- **Bundled-trained-artifact discipline (ADR-011/016/040/044/050)** for `PassCompletionModel`: pickle-free
  JSON + `SHA256SUMS`, feature contract, chirality probe, fail-closed load, inference imports no sklearn.
- **Canonical-id grouping** (ADR-019), **drop-and-count conservation** (ADR-042), **ADR-028** reflection,
  **purity** (ADR-033), **artifact provenance** (`scripts/_provenance.py` `require_clean_tree` +
  `run_commit`/`training_commit`; ADR-052 `for_each` shards; ADR-056 input contract).
- **`for_provider` ships EMPTY** (ADR-009) for `TerritoryParams`, `CounterfactualParams`, and the
  completion-model per-provider hook. Census/gate thresholds are **locked constants committed before the
  run** (the TF-19 `NAMED_KEEPER_PRIOR` idiom).

## 4. Architecture overview

```
silly_kicks/expected_passing/            # ALREADY ON MAIN (carried from ab9001c)
  __init__.py / _model.py / _features.py / weights/   # PassCompletionModel; weights RE-FIT + RE-BUNDLED this cycle
silly_kicks/xthreat/
  _counterfactual_seam.py                # ALREADY ON MAIN — destination_profiles (cone-restricted distribution)
silly_kicks/territory/
  _columns.py            # + per-method column/dtype resolver + counterfactual-only columns   [REVIVE]
  _config.py             # + CounterfactualParams (frozen)                                     [REVIVE]
  _report.py             # + counterfactual census fields (conservation)                       [REVIVE]
  _compute.py            # RE-ADD dispatch: completed_failed (unchanged) | counterfactual      [REVIVE]
  _counterfactual.py     # NEW — the joint valuation q x c x xT                                [REVIVE]
scripts/
  train_pass_completion.py               # ALREADY ON MAIN — re-run on the ~3,961 corpus this cycle
  validate_territory_counterfactual.py   # NEW — owner-run construct-validity corpus pass       [REVIVE]
  _synthetic_interception.py             # NEW — synthetic-interception target-recovery substrate [REVIVE]
  build_territory_ranking_census.py      # NEW — crossed defender+team-cell census + ICC gate   [NEW, this cycle]
docs/research/territory_counterfactual_construct_validity/   # NEW — validation artifact (Commit 2)
docs/research/territory_ranking_census/                      # NEW — census + ranking artifact (Commit 2)
```

`territory.build_trimmed_hull`, `xthreat.destination_profiles`, `expected_passing.PassCompletionModel`,
`scripts/_sb_open_data` are **already on `main`** (verified importing clean) — the revival is the cone
consumer + dispatch + validator + census, not the whole branch. C4: the `expected_passing` container is
already modelled; the **33 action-coupled `add_*` aggregator count is unchanged** (neither the
counterfactual method, the completion model, nor the census is an `add_*`).

## 5. Feature design — `method="counterfactual"` (settled baseline, ported verbatim)

The quantity, the three components (`q` × `c` × `xT`), membership rules, the method-dependent schema, the
worked-example golden, `CounterfactualParams`, and windowing pooling-by-kind are **exactly** the
R2-approved baseline (`2026-09-05-…-design.md` §5). Reproduced here for the reviewer; unchanged.

### 5.1 The quantity
Unified expected − realized, summed over opponent passes **aimed into** the territory.
`territory_xt_conceded` = realized threat of completed passes; `territory_xt_prevented` =
`Σ_{failed aimed-in} Σ_z q·c·xT`; headline `territory_xt_prevented_above_expectation` =
`Σ (P_complete − outcome)·xT(target)`.

### 5.2 The three components
- **Target distribution `q` (Tier-2, injected xT).** For a failed pass from origin cell `i` with observed
  death direction `θ`: `q(z) ∝ T[i, z]` over territory zones `z ∈ R` in the cone of half-angle
  `direction_cone_degrees` around `θ`, renormalized. Sourced through `xthreat.destination_profiles` (public
  seam), used only as a **distribution** (family-agnostic — valid for both `singh_counts` and
  `kde_smoothed`). Death-direction is a noisy proxy (an interception's death is where the ball was
  redirected), which is why `q` spreads mass over a cone rather than a ray; the synthetic-interception test
  perturbs both distance and angle.
- **Completion `c` (`PassCompletionModel`).** `c(origin, z) = P(complete | origin→z geometry)`, evaluated
  at each hypothesized target `z` (failed) or the observed end (completed).
- **Value `xT(z)` (injected xT).** `values_at_points` at each `z`.

**Composition.** Failed: `expected = Σ_{z∈R∩cone} q(z)·c(origin,z)·xT(z)`. Completed:
`expected = c(origin,end)·xT(end)`, `realized = xT(end)`, contribution `(c(origin,end) − 1)·xT(end)`. A
pass with no cone∩R transition support (`Σ T = 0`) or non-finite completion features → **unresolvable**,
dropped-and-counted (`territory_target_source="unresolved"`), never a fabricated 0.

### 5.3–5.5 Membership, schema, params, windowing
- **Completed** counts iff observed `end` is point-in-hull (v1 rule → `xt_conceded` value formula-identical
  to v1). **Failed** counts iff its death-direction cone from the origin intersects the reflected hull
  region `R`.
- Method-dependent columns (counterfactual-only): `territory_expected_threat_faced`,
  `territory_xt_prevented_above_expectation`, `territory_passes_aimed_into_hull`,
  `territory_mean_completion_faced`, `territory_target_source`. `completed_failed`'s shape/values untouched.
  Rate denominators become `passes_aimed_into_hull` (documented method-dependent). Each column glossaried
  with `higher_is_better`; no two columns silently coincide.
- `CounterfactualParams` (frozen; `.default`/`.for_provider`/`.is_default`): `direction_cone_degrees`,
  `min_transition_support`, `target_zone_grid` (defaults to `xt.grid`). `_PROVIDER_COUNTERFACTUAL_PARAMS`
  ships **empty** (ADR-009).
- Windowing pools by kind (sums add; rates re-derived from pooled sums; `mean_completion_faced` is
  support-weighted; `target_source` dropped/ summarized at window grain).
- The worked-example golden (§5.7 of the baseline) is ported as the core hand-checkable test: totals
  `conceded=0.15`, `prevented=0.078`, `expected_threat_faced=0.168`,
  `xt_prevented_above_expectation=0.018`, plus the uniform-xT invariant `prevented=0.06` exactly.

## 5b. `PassCompletionModel` — re-fit + re-bundle (delta 3)

The model itself is unchanged (logistic, pure-numpy `sigmoid(Xβ)` serve, pickle-free JSON + `SHA256SUMS`,
feature contract + chirality, fail-closed load, event-only features; a GBM variant remains a reserved door
decided by held-out validation). **This cycle re-fits it on the full public open-data corpus
(`_sb_open_data.all_open_competitions()`, ~3,961 matches — estimate, re-counted at run; §1)** and re-bundles the weights, replacing the
WC2022-trained default currently on `main`. Rationale: a broader default is a better default and matches
the substrate the metric is validated on. The re-fit is owner-run at Commit 1 (clean tree, `--out` outside
the repo), `training_commit` = Commit 1; the model card records the corpus change. Held-out AUC/ECE/Brier
(GroupKFold-by-match) must clear the pre-registered floors (mirroring `GkRetentionModel`'s
`ece ≤ 0.10`-style gate) or the re-bundle does not ship (fall back to the existing bundled weights, recorded).

## 5c. The public xthreat seam (already on main)
`xthreat.destination_profiles` returns, for an origin point and a zone set, the renormalized destination
distribution over those zones plus their physical centres and `xT` values, with flat-index + y-inversion
owned inside `xthreat` and family-agnostic. `territory` consumes only this seam. Already landed; the
revival adds the `territory`-side consumer, not the seam.

## 6. Rejected alternatives (inherited + revival-specific)
Inherited from the baseline (drop the completion weight; source `P_complete` from `T`; fail-closed on a
non-singh xT; evaluate `c` at death geometry; possession-value ghosting; same-columns-changed-values) —
all still rejected for the reasons recorded in `2026-09-05-…-design.md` §6. Revival-specific:

| Option | Why rejected |
|---|---|
| **Restrict validation to WC2022** (the baseline's public leg) | The point of the revival is broad-corpus; ~3,961 open matches are available in-repo and give the multi-season club-league structure the ranking needs. WC2022-only forecloses the ranking by construction. |
| **Keep the WC2022-trained bundled weights** | Owner decision 2026-09-17: re-fit on the broad corpus (better default; validated substrate == serving substrate). |
| **Ship a defender ranking unconditionally** | The player-vs-team confound is a statistical-validity constraint, not a scope cut; "scope not a concern" cannot license an unidentifiable ranking. Gate it on the crossed-cell census (§8). |
| **Add a ranking function to the library** | ADR-009 keeps rankings consumer-side; the library ships the per-`(defender, match)` primitive, the ranking is a reported artifact. |
| **Re-open / re-review the cone mechanism DESIGN** | It reached R2 APPROVE (a design verdict); the revival treats the *design* as settled and spends its review budget on the deltas. This is NOT a validation claim — the mechanism was never run (§1 SPEC-01); its empirical Primary-1/2 are what THIS cycle produces, and a null is reportable. |

## 7. Validation design (broad corpus; pre-registered, in-cycle, reported-not-gated)

### 7.1 Substrate & drivers
- **Primary public leg:** the full StatsBomb open-data corpus via `_sb_open_data.all_open_competitions()`
  (~3,961 matches — estimate, re-counted at run (§1); all open competitions/seasons;
  `assert_statsbomb_open_data_mode()` fail-closes against the private API). Optional owner-tier
  cross-check: Gradient Sports WC2022 via pining.
- `train_pass_completion.py`, `validate_territory_counterfactual.py`, `build_territory_ranking_census.py`
  are `for_each`-sharded (ADR-052), `require_clean_tree` + provenance-stamped (ADR-037), input-contract
  declared (ADR-056). xT and the completion model used for the validation are fit on a corpus **disjoint**
  from the scored matches (the validator fits its own leakage-disjoint models; it does not depend on the
  bundled weights, so its provenance is clean).

### 7.2 Pre-registered test battery
**Component-level (true ground truth):**
- `PassCompletionModel` — held-out AUC / ECE / Brier across the geometry range (probing the failed-pass
  hypothesized-target interpolation), GroupKFold-by-match, pre-registered floors.
- Target recovery (`q`) — synthetic-interception substrate (`scripts/_synthetic_interception.py`): take a
  completed pass (true end = ground truth), synthesize an interception at flight-fraction `f∈(0,1)` along
  origin→end **and** a random angular offset `δ` off the ray, hide the true end, require the
  direction-conditioned estimator to recover it better than (a) "death = the synthesized intercept" and
  (b) origin-zone centroid. Leakage-free + discriminating; the composed Primary-1 also runs the held-out
  real-data leg (genuine interception noise).

**Composed-metric primary (decides default promotion — a separate ADR-009, both must clear):**
1. **Mechanism validity** — the composed counterfactual (validated `c` and `q`) beats the v1 death-location
   proxy and the naive baselines on the synthetic-interception + held-out substrate.
2. **Face validity — the "Van Dijk" prior** — a locked, pre-registered elite-defender list (committed as a
   constant in Commit 1, stamped in the artifact) surfaces in the top decile/quartile at meaningful
   pass-faced volume, **measured on the broad corpus** (where the elite club defenders actually appear).

**Composed-metric secondary (reported, not gating):** reliability (split-half / across-window),
discriminant (vs v1 `completed_failed`, vs volume, vs team defensive strength; beats a shuffled-outcome
placebo; includes the decomposition-non-degeneracy check that the split is not ≈`−conceded`), outcome-lens
(possession-reaches-shot AUC, explicitly not gating).

## 8. Defender-ranking census + gate (delta 4 — the new deliverable)

The per-`(defender, match)` metric is team-conditioned by construction (a defender's blunting is entangled
with the team defending around them). A **ranking** across defenders is therefore licensed only if the
corpus can separate the defender-intrinsic component from the team component — a crossed **defender+team**
random-effects identifiability question (the ADR-090 posture, made testable rather than assumed).

`scripts/build_territory_ranking_census.py` (`for_each`-sharded, provenance-stamped, event-only) runs in
two tiers with **locked pre-registered thresholds** committed in Commit 1:

- **Tier 1 — counting census (always runs).** From public lineups + the metric's per-`(defender, match)`
  rows: distinct defenders; **defenders appearing on ≥2 distinct teams** (the transfer-driven crossed
  cells the multi-season club leagues provide); defender×team cells clearing a `min_passes_faced` volume
  floor; the resulting design's coverage. Reported unconditionally (`territory_ranking_census/`).
- **Tier 2 — crossed defender+team ICC + power (runs iff Tier 1 clears its minimum).** A pure-numpy crossed
  random-effects variance-components estimate (method-of-moments / ANOVA decomposition; no statsmodels,
  no PyMC) of the defender-intrinsic ICC net of team, plus a bootstrap power leg. **Both are NEW work
  (SPEC-03):** `causal.power` ships **only** `att_power_curve` (an ATT estimator) — there is **no**
  `icc_power_curve` anywhere in the library (verified). So neither
  the crossed variance-components estimator nor its bootstrap power leg is a reuse; both are implemented
  here (the bootstrap may borrow `causal.power`'s resampling *style*, not a curve). Pre-registered gate:
  the crossed defender ICC is estimable (lower CI bound > 0) **and** the design is powered (bootstrap power
  ≥ the locked floor at the locked effect size).

**Decision rule (owner-approved 2026-09-17):**
- **Gate clears** → publish the **ranking artifact** in `territory_ranking_census/` (top defenders by
  `xt_prevented_above_expectation` at adequate pass-faced volume, with the crossed-ICC evidence) **and**
  record an ADR-009 apply that the metric is player-attributable **on this corpus**. The per-`(defender,
  match)` metric columns and the library surface do not change.
- **Gate fails** → the cycle ships the metric only; the census artifact reports "ranking not licensed on
  this corpus" with the measured counts/ICC/power, and the ranking stays a further ADR-009 gated on a
  larger multi-club transfer corpus (unchanged posture).

Either way the census runs, is reported, and conserves its counts (ADR-042). The gate is the honest
guard: it cannot be satisfied by volume alone (Tier 1's ≥2-team requirement is the identifiability
precondition, not a match count).

## 9. Testing (two tiers)

**Regular suite (`-m "not e2e"`, committed fixtures):**
- `PassCompletionModel`: fit/predict on a toy fixture; pure-numpy serve == training-lib prediction;
  JSON+SHA256 round-trip; load guards (chirality mismatch raises; missing/altered feature contract raises,
  ADR-050/044); a hand-computed logistic value; NaN-feature → NaN/`unresolved`. Event-only import-allowlist
  with planted-violation meta-tests. (Model already on main; tests extended for the re-fit template only if
  the feature set changes — it does not.)
- The xthreat destination-profiles seam: already tested on main; no new seam tests (revival consumes it).
- `_counterfactual.py`: the joint `Σ q·c·xT` against hand-computed values on toy `xt` + toy `c`;
  **`completed_failed` byte-identity** (the load-bearing additive guard); method dispatch + method-dependent
  schema; membership (completed point-in-hull; failed cone∩R; aimed-in-died-short counted;
  unresolvable-target dropped-and-counted with report conservation); the ported worked-example golden.
- Reflection-invariance (per-row; one scene from either perspective scores identically), incl. a failed
  pass whose death ≠ intended target.
- Mechanism-recovery-beats-naive on a fixture (the synthetic-interception offline half).
- **Census (new):** the crossed variance-components estimator against a hand-constructed fixture with known
  defender/team variance (recovers the components); the ≥2-team counting logic; Tier-1→Tier-2 gating
  (a corpus with no crossed cells stops at Tier 1); conservation of census counts. The gate thresholds are
  read from the locked constants, and a from-both-sides test asserts a below-threshold design does NOT
  license the ranking.
- Purity (ADR-033, ≥2 variants); territory + expected_passing import-allowlists; glossary coverage for new
  columns; NOTICE citations; C4 completeness.

**e2e (`@e2e`, owner-run):** the full corpus training + validation + census (§7–§8) producing the artifacts.
Note (SPEC-06): only the synthetic-interception **offline** half of Primary-1 has a committed regression
(above); the **composed real-data** Primary-1/2 and the census ICC are owner-run e2e by nature (they need
the corpus), so their result — including a possible null (§1 SPEC-01) — lives in the Commit-2 artifacts,
not in CI.

## 10. Commit plan

**One feature branch off `main`** (`feat/tf54b-counterfactual-territorial-prevented-revival`).

- **Commit 1 — all code + release (clean tree).** `territory/_counterfactual.py` + re-added dispatch +
  method-dependent `_columns.py` + `CounterfactualParams` + `_report.py`; the locked elite-defender prior
  constant + the locked census/ICC thresholds; `scripts/validate_territory_counterfactual.py` +
  `scripts/_synthetic_interception.py` + `scripts/build_territory_ranking_census.py`; all offline tests
  (§9, using fixture-fit models, not bundled weights); `feature_glossary` + `glossary_emitted_columns` +
  `NOTICE`; C4 `dot` re-render (33 aggregators unchanged); version bump; `CHANGELOG`; **ADR-NNN**. Fully
  green: `-m "not e2e"` + ruff + ruff format + bare pyright. **Default stays `completed_failed`.** C4 is
  unchanged (the `expected_passing` container is already on `main`; no new container/aggregator/backend/
  model — re-render only if a structural element is actually added).
- **Owner-run at Commit 1 (clean tree, `--out` outside the repo):** (a) re-fit `PassCompletionModel` on the
  ~3,961 public corpus → re-bundled weights (`training_commit` = Commit 1); (b) run the
  territory-counterfactual validation pass (fits its own leakage-disjoint models) → `metrics.json` +
  `named_defender_signs.parquet` + `findings.md`; (c) run the ranking census → `census.json` +
  (iff the gate clears) `ranking.parquet` + `findings.md` (`run_commit` = Commit 1).
- **Commit 2 — the provenance artifacts** (re-bundled `expected_passing/weights/` JSON+SHA256, the
  `territory_counterfactual_construct_validity/` report, the `territory_ranking_census/` artifact), all
  stamping Commit 1. The single justified second commit (committing artifacts with the code would force a
  dirty-tree stamp, which provenance discipline forbids). Load-bearing non-squash so the stamps resolve.

Then owner-driven, explicit approval at every gate (never a commit/push/tag without an unambiguous yes):
push → PR → CI green → admin-merge non-squash → tag → PyPI.

**Version / PR-Snnn locked only at Commit 2** (owner rule). At commit-prep `git fetch && git merge
origin/main`, then take the next free number from `main` — another session's work may renumber. Nothing
in this doc pins the number.

## 11. Open / plan-time details

- The crossed variance-components estimator: the exact method-of-moments decomposition + its bootstrap
  power leg (NEW — `causal.power` has no ICC curve; see §8 SPEC-03), and the locked ICC floor / power floor
  / effect size.
- The `min_passes_faced` census volume floor and the ≥2-team Tier-1 minimum (locked constants).
- The elite-defender prior list contents for the broad corpus (locked before the run) + top-quantile
  threshold — the broad corpus reaches club defenders the WC2022 list could not, so the list is re-drawn.
- `PassCompletionModel` re-fit floors (AUC/ECE/Brier) and the fallback-to-existing-weights recording path.
- Confirm the failed-pass hypothesized-target geometry stays within the re-fit model's validated range on
  the broad corpus.
- `_report.py` counterfactual + census census fields (`n_target_modeled`, `n_target_unresolved`,
  `n_defenders`, `n_multi_team_defenders`, …) + conservation.
- Redistribution check: the re-bundled weights are public-corpus-trained (in-repo JSON; confirm no licensed
  data via `assert_statsbomb_open_data_mode`).

## 12. References (NOTICE)
- Singh, Karun. "Introducing Expected Threat (xT)." 2019 — the transition/value engine.
- Sumpter, D. *Soccermatics* / Twelve module 10.2 (the "Van Dijk" territorial-dominance lens) — TF-54.
- Expected-passing / pass-completion modelling (field-standard "expected pass") — `PassCompletionModel`.
- The GSAA "expected − realized" framing — the house's TF-59 shot-stopping (ADR-085) analog.
- Counterfactual defensive valuation (ghosting: Le et al. 2017) — cited as the reserved comparator, not
  implemented.
- Intraclass-correlation identifiability of player vs team effects — the ADR-090 confound (the crossed
  variance-components + bootstrap power leg are new; `causal.power` provides only `att_power_curve`).

## 13. Decision log
- **2026-09-05 (baseline, R2 APPROVE):** the cone mechanism — Option D completion-weighted expected −
  realized; `q` from the injected xT (family-agnostic distribution), `c` from a new `PassCompletionModel`,
  value from xT; new public xthreat seam; in-cycle reported-not-gated validation; method-dependent schema;
  `completed_failed` byte-identical. (Full log in `2026-09-05-…-design.md` §12.)
- **2026-09-17 (revival brainstorm, owner):** revive Path B onto current `main` (seams already landed;
  door removed → re-add); **broad public open-data corpus** (~3,961 via `_sb_open_data`) not WC2022;
  **re-fit + re-bundle `PassCompletionModel`** on that corpus; **defender-ranking census + crossed-ICC gate**,
  ship the ranking artifact **in-cycle iff the gate clears** (else metric-only + reported census); ranking
  stays a reported artifact + ADR-009 apply, not a library API; fresh **ADR-NNN** (ADR-089 collides,
  ADR-090 is the tracking sibling). Scope/breaking cleared by owner; gold-standard bar.
