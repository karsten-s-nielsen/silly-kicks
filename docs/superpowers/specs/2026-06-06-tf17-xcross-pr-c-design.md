# TF-17 PR-C: xCross Causal Validation Harness — design

**Date:** 2026-06-06
**Status:** approved (brainstorm), pre-implementation
**Parent design:** `docs/superpowers/specs/2026-06-03-tf17-xcross-attempt-design.md` (§9 causal harness, §10 PR-C scope, §11.3 port unit tests, §11.4 e2e). This document is the dedicated, consolidated PR-C spec; it does not re-litigate the owner-reviewed §9 decisions, it makes them buildable and records the one refinement agreed in the PR-C brainstorm (the pure `opportunities.py` split).
**Decision record:** ADR-015 (lands with this PR).
**Closes:** TF-17 (final PR of the A/B/C split — PR-A code 4.11.0, PR-B weights 4.16.0).

---

## 1. Purpose

PR-C is the **paper-faithful causal validation arm** of TF-17. The runtime surface (PR-A/PR-B) answers
*"will the in-possession team cross from this state?"* (a STATE-anchored propensity surface). PR-C answers a
different, causal question, faithful to Cao et al. (2025, arXiv:2505.11841): **is goalkeeper position a real
backdoor confounder of the cross → shot-creation effect?** It does this with propensity-score matching on
**crosser-anchored opportunity rows**, ablating the novel GK-confounder block against a feature-matched
placebo null band.

This is a **standalone research artifact**. It does **not** gate PR-B's weights or TF-19 — a null causal
finding is a valid result, not a regression (H-2/H-3 from the parent design). Only the known-truth *method*
unit tests gate CI.

### Why it matters now (motivation strengthened by PR-B)

PR-B shipped with `tf19_ready = False`: the GK substitution-sensitivity probe moved P(cross) by a median
0.00107 — 2.6× the nearest-defender control but **below** the pre-registered 0.01 absolute floor. The surface
GK signal is weak. The causal harness is the **independent, paper-faithful test** of whether GK is nonetheless
a real cross→shot confounder despite weak surface movement — exactly the question a weak-but-nonzero surface
signal leaves open.

---

## 2. Estimand & DAG (parent §9.1, settled)

Treatment `Z` = a cross is attempted by the possessing team from a crossing opportunity. Outcome `Y` = the
possessing team creates a shot within a short post-window. Assumed DAG:

```
            attacking state S
           /       |        \
          v        v         v
   GK_position    Z (cross)   Y (shot)
       |  \________^          ^
       |           (deterrence: GK_pos -> Z)
       \___________________________________> Y   (GK_pos -> shot prevention)
```

- `S` is the confounder the paper proxies via its 8 covariates (`S -> Z`, `S -> Y`, `S -> GK_position`).
- **GK_position is a pre-treatment backdoor confounder, not a mediator or collider** — measured at frame `t`,
  the cross decision occurs in `[t, t+horizon]`, so the GK measurement strictly precedes treatment.
  Conditioning on a pre-treatment common cause of `Z` and `Y` *reduces* confounding bias. The harness states
  this DAG and the pre-treatment guarantee explicitly; a variant measuring GK position *after* the cross would
  turn it into a mediator and invalidate the analysis.

### Faithfulness caveat (recorded in NOTICE)

The paper's treatment is **sender-level** (one row per crossing opportunity, anchored to the would-be
crosser). The runtime surface is **state-level per-frame**. PR-C deliberately reconstructs the **sender-level**
unit for the causal test (§4), so the causal harness *is* faithful to the paper's unit — distinct from the
runtime surface. The remaining unavoidable divergence (tracking-only opportunity detection vs the paper's
event-stream opportunity labels; different league/era corpus) is logged, not hidden.

---

## 3. Module layout

```
silly_kicks/_causal/
  __init__.py            # private package; exports nothing public (promote to silly_kicks/causal/ only
                         #   when a 2nd consumer — TF-19 — actually lands; ADR-015 records this)
  matching.py            # pure estimator port (no I/O, provider-agnostic)
  opportunities.py       # pure crosser-anchored opportunity-row builder (the PR-C-brainstorm split)
scripts/
  validate_xcross_causal.py   # thin I/O driver over the port; produces metrics.json + report
tests/causal/
  __init__.py
  test_matching.py       # known-truth unit tests (regular suite, NOT e2e)
  test_opportunities.py  # opportunity-build + dedup unit tests (regular suite, NOT e2e)
  test_causal_e2e.py     # @pytest.mark.e2e: harness-runs-and-reports
docs/superpowers/adrs/ADR-015-causal-validation-port.md
```

**`silly_kicks/_causal/` is private** (leading-underscore package), mirroring `_xcross_eval.py`'s
private-research-surface precedent. It is **not** imported by `silly_kicks/__init__.py` and ships no public
API. `import silly_kicks` stays dependency-light: `matching.py` imports only numpy + `sklearn.neighbors` /
`sklearn.linear_model` (both already runtime deps).

### 3.1 `matching.py` — pure estimator port (parent §9.3)

Pure functions, each numpy-in / numpy-out, deterministic given a seed:

| Function | Contract |
|---|---|
| `fit_propensity(X, Z, *, seed)` | Logistic regression (sklearn `LogisticRegression`, fixed `C`/`solver`) on **standardized** confounders `X` → propensity `e(x) ∈ (0,1)`. **Covariates are z-scored before the fit** (M6 — the raw confounders are multi-scale: metres, radians, counts; lbfgs is not scale-robust). Returns scores + fitted coefs. |
| `propensity_match(ps, Z, *, target)` | 1:1 nearest-neighbor matching on the propensity score, **with replacement**, ties allowed, **no caliper** (paper-faithful). `sklearn.neighbors.NearestNeighbors` on the control pool for treated (ATT) and the treated pool for controls (ATNT). NN tie-break is index-order deterministic — **no `seed` param** (L1). Returns `{focal_idx: matched_idx}`. |
| `estimate_att(Y, Z, ps, X)` | ATT = mean over treated of `Y_i − Y_{m(i)}`. Deterministic given `ps` (no seed). Returns a `CausalEstimate(estimate, se, balance, n_focal, matched)`. |
| `estimate_atnt(Y, Z, ps, X)` | symmetric (effect on the untreated). |
| `smd_balance(X, Z, matched, *, target)` | standardized mean differences per covariate, pre- and post-match; returns a table (`covariate`, `smd_pre`, `smd_post`). |
| `abadie_imbens_se(Y, Z, matched, ps, *, target)` | Abadie–Imbens (2006) matching variance estimator — `var = (Σ(τ_i−τ̄)² + Σ_j K_j(K_j−1)σ̂²_j) / N₁²`, the Imbens & Rubin (2015, Ch. 19) ATT variance under matching-with-replacement (naive and bootstrap SEs are biased here). Deterministic (no seed). `σ̂²(X_i)` via the J=1 within-treatment-group NN (a documented approximation; see ADR-015). **NB:** matching is on the *estimated* PS — the naive formula is conservative (Abadie–Imbens 2016); acceptable for a *reported* artifact, named in ADR-015. |
| `placebo_shift(X_base, X_gk, Y, Z, *, n_seeds, rng_seed)` | the GK-ablation null: the placebo block is the **row-permuted GK block** (shuffle the row order of `X_gk` per seed — preserves the GK block's marginals + within-block correlation, destroys its alignment with `Z`/`Y`; H3). **Note (R2-L3):** permuting rows also breaks GK↔`X_base` correlation, so the null is *slightly conservative* vs a pure `Z`/`Y`-alignment null (standard for permutation nulls; documented in ADR-015). Returns `{"shifts", "band_p95", "base_att"}`. |

All randomness flows through explicit `seed`/`rng_seed`; no `np.random` global state — reproducible (mirrors
the deterministic-XGBoost discipline elsewhere).

### 3.2 `opportunities.py` — pure crosser-anchored opportunity builder (parent §9.2 + brainstorm split)

**The PR-C refinement:** the §9.2 spell-construction logic is its own pure module, not buried in the script,
so the dedup rule (R2-M1) is directly unit-testable (`test_opportunities.py::test_opportunity_dedup`).

- **Unit:** a *crossing opportunity* = one **continuous wide-area possession-spell** — a maximal run of
  consecutive in-possession frames (same possessing team, no turnover, ball continuously inside the
  wide-area/advancement domain), anchored to **the carrier at spell entry**. The builder is a **spell
  state-machine** that tracks spell *entry* and *end*; the end serves as the dedup boundary (one row per
  continuous spell; `MAX_SPELL_SECONDS` caps a pathological never-closing run) **and** as the ceiling on the
  treatment window (R3-M1, below). It emits one row per spell, recording `entry_time`, `end_time`,
  `spell_duration_seconds`.
- **Treatment — fixed-cap window clamped to possession (R2-H3 + R3-M1):** `Z=1` iff a cross by the possessing
  team occurs in **`(entry_time, min(entry_time + EXPOSURE_WINDOW_SECONDS, spell_end)]`**. Two caps, two
  purposes: (1) the **fixed `T` cap** gives every opportunity the same *maximum* Z-exposure → no spell-length
  confounding (the rev.2 variable `[entry, end]` window made `spell_duration` a common cause of `(Z, Y)`); (2)
  the **`spell_end` cap** ties Z to *this* uninterrupted possession → no misattributing a cross from a later
  re-possession phase (rev.3's pure-fixed-window cost). Crucially the clamp introduces **no new duration
  confounding** because `Y`'s window is *already* fixed (next bullet), so varying the Z at-risk window does not
  create a duration→`Y` exposure path. `T` named + pre-registered; `spell_duration_seconds` reported as a
  diagnostic. (Owner decision, round-3: Option B — clamp to possession — over the pure fixed window.)
- **Outcome — anchored strictly post-treatment (R2-M1):** to avoid immortal-time / reverse-direction leakage
  (a shot *before* the cross counting as `Y`), `Y` is measured over a **fixed-length** window
  `OUTCOME_WINDOW_SECONDS` (`W`) from a per-unit anchor: **treated → `(t_cross, t_cross + W]`** (strictly after
  the cross), **control → `(entry_time, entry_time + W]`**. `t_cross` = the first cross time in the Z-window.
  Same window length `W` for both arms → equal Y-exposure (no exposure confounding); treated `Y` strictly
  post-cross → no anti-causal contribution. The treated-anchored-later asymmetry is a documented modeling
  choice (the cross is the treatment; its effect on a *subsequent* shot is measured from the cross). `W` named +
  pre-registered.
- **Confounders** `X` = the **7 paper confounders** (M3) — `score_differential`, `dist_nearest_def`,
  `space_controlled`, `dist_nearest_teammate`, `dist_endline`, `box_off_def_ratio`, `ten_minute_warning` —
  extracted at spell entry via `extract_xcross_features`. **The list is imported from
  `_xcross_attempt._CONFOUNDERS`, not re-literal'd (R2-M2)** — single source of truth, so the "faithful 7"
  cannot silently diverge if the paper set is revised upstream (`PAPER_CONFOUNDERS = _CONFOUNDERS`).
  **The 3 ball-geometry features (`ball_r/theta/speed`) are deliberately EXCLUDED from the causal propensity
  model** — they are surface-model inputs, not paper confounders; including them would diverge from the
  paper-faithful confounder set (the stated reason this PR exists). The **GK block** (`gk_*`, 6 cols) is the
  separately-toggleable group ablated in §4 (isolatable per the shipped `test_gk_block_isolatable`).
- **`score_differential` IS populated (M1):** the builder wires `_ghost_gk._build_score_lookup(actions,
  home_team_id)` and passes `score_differential` into `extract_xcross_features` (mirrors
  `prepare_xcross_training_data`), so this paper confounder is real, not silently NaN. `build_opportunities`
  therefore takes `actions` + `home_team_id`.
- **GK missingness is a signal, not noise (M2):** a NaN GK column means *GK not detected / occluded* —
  mean-imputing it would attenuate exactly the association the harness exists to detect. The driver uses the
  **missing-indicator method** (a `gk_missing` binary covariate + imputation of the GK columns) and surfaces
  **both** `gk_nan_fraction` **and** `base_nan_fraction` (the 7 paper confounders are mean-imputed for the PS
  fit, but their NaN fraction is reported too — R2-M3) loudly in `metrics.json`.
- **Dedup rule (R2-M1):** re-entering the corridor after exiting (or after a turnover) starts a **new** spell
  only if the possession broke or the ball left the domain; a carrier hand-off *within* one continuous
  in-domain spell does **not** create a second opportunity (one row, anchored at entry).
- **Domain definition matches the trained model's, not re-literal'd.** `cross_types` and `carrier_params` are
  read from the bundled `default` `metadata.json`; the **wide-area domain predicate** itself is *reused* from
  `_xcross_attempt.py` (`_in_wide_area` + `_ADVANCE_M` + `_build_goal_map`), the same predicate
  `prepare_xcross_training_data` applies (DRY) — so the matched corpus is the same domain the model was trained
  on by construction. (`horizon_seconds` from metadata governs the *surface* model, not the spell window — see
  the Treatment note above.)

`build_opportunities(frames, actions, *, home_team_id, model_metadata, ...) -> pd.DataFrame` returns one row
per closed spell with the 7 confounder cols + the 6 GK cols, `Z`, `Y`, plus provenance (`game_id`,
`period_id`, `entry_frame_id`, `entry_time`, `end_time`, `spell_duration_seconds`, `possessing_team`,
`provider`, `carrier_resolved`). Pure; no I/O.

### 3.3 `scripts/validate_xcross_causal.py` — driver (pure `analyze()` + I/O `run()`)

Split into a **pure `analyze(opp_df, *, seed, ...) -> metrics_dict`** (all estimation/orchestration, no I/O —
so the e2e can drive it on a built opportunity frame) and a thin **`run(...)`** that does only loader I/O +
`analyze` + artifact write (H2). `analyze` does:

1. Split `X` into the 7 paper confounders (`X_base`) + the 6 GK columns (`X_gk`); apply the **missing-indicator
   method** to `X_gk` (M2 — add `gk_missing`, then impute); record per-block NaN fraction.
2. **Positivity guard (M5):** if `n_treated == 0` or `n_control == 0` (e.g. a team-id-space mismatch on a new
   provider), **do not** silently produce `nan` — set `metrics["status"] = "no_variation_in_treatment"` and
   stop (loud, surfaced).
3. Headline experiment (§4): ATT without GK vs with GK; the row-permuted-GK placebo null band; AI SEs.
4. **Overlap + balance gate (M4):** compute a PS common-support/overlap diagnostic (fraction of treated inside
   the control PS **range** — a range check, **not** density trimming; the report says so, R3-L4) AND
   `max|smd_post| < max|smd_pre|`. If overlap fails OR balance does not improve, set
   `metrics["causal_claim_supported"] = False` and say so loudly — **no causal claim is made** (no-silent-caps,
   parent §9.4c). Both pre- and post-match SMD are reported, not just post.
5. Sign sanity vs the paper (positive = cross → more shots; logged, not gated).

`run(...)` then: load frames+actions per provider (`scripts/_loader_pining.py`, PR-B corpus selection) →
`build_opportunities` per match → concat → **carrier-coverage validity (L-2):** per-provider carrier-resolution
coverage reported; **low-coverage providers excluded from the headline ATT** (pre-registered threshold; a
selection bias, not just weaker features) → `analyze` on the eligible pool → write `metrics.json` + a short
markdown report.

---

## 4. Headline experiment — GK ablation with a feature-matched placebo (parent §9.4)

Report **ATT** with vs without the GK block. **An ATT shift alone is NOT evidence** — adding any equal-width
covariate block perturbs a matched ATT via overlap loss, propensity-model variance, and finite-sample
matching jitter. The pre-registered protocol (mirrors the trained-model-validation-rigor discipline):

- **(a) Placebo null band (H3):** the same ablation with the **row-permuted GK block** as the placebo — shuffle
  the row order of the actual `X_gk` matrix per seed. Permutation **preserves the GK block's marginal
  distributions and within-block correlation** and destroys only its alignment with `(Z, Y)`, so the null
  isolates "GK carries `Z`/`Y` signal" rather than "GK columns aren't unit-Gaussian" (a pure random-normal
  block would conflate the two). Across seeds → the null distribution of ATT shifts. The GK shift is "real"
  **only if it clears the placebo band** (the 95th percentile of `|shifts|`). This is the feature-matched null,
  not an unconditional zero.
- **(b) Pre-registered effect-size threshold** for "non-trivial," fixed in code before the run (named
  constants in `matching.py`, unit-asserted), mirroring PR-B's `TF19_PROBE_*` pins.
- **(c) SMD balance must improve post-match and overlap/common-support must hold;** no causal claim is made
  if overlap or balance fails (no-silent-caps — the report says so loudly).

**Reported, never a ship gate or CI assertion.** A GK shift that does not clear the placebo band is a valid
null. The e2e test asserts the harness *runs and reports* both numbers, never that GK wins.

---

## 5. Testing (TDD — tests authored before implementation)

### 5.1 `tests/causal/test_matching.py` (regular suite, NOT e2e) — known-truth

- `test_recovers_known_ate` — synthesize data with a known ATE + known confounding; `estimate_att`/`atnt`
  recover it within tolerance.
- `test_smd_balance_improves_post_match` — SMD on confounders strictly drops after matching.
- `test_placebo_block_zero_shift` — a permuted/random block yields ~zero ATT shift (the null GK must beat).
- `test_with_replacement_reuses_controls` — a control can match multiple treated (structural check).
- `test_abadie_imbens_se_positive_finite` + a constructed case where AI SE ≠ naive SE (documents why naive is
  wrong under replacement).
- `test_no_caliper_keeps_all_treated` — no treated unit dropped (paper-faithful).
- `test_propensity_deterministic` — same seed → identical matches (reproducibility pin).

### 5.2 `tests/causal/test_opportunities.py` (regular suite, NOT e2e) — dedup is the unit under test (H4)

The spell state-machine is the subtlest logic in the PR; each R2-M1 branch gets a **distinct, non-trivial**
fixture (no byte-identical no-op tests):

- `test_single_spell_one_row` — one continuous wide-area team spell → exactly one row, anchored at entry.
- `test_reentry_after_turnover_is_new_spell` (R2-M1 positive path) — team A spell → **turnover to B** → A
  regains in the corridor → **2 rows** (a genuine second-spell fixture, not just a trailing opponent frame).
- `test_reentry_after_domain_exit_is_new_spell` — same possession, ball **leaves the wide area** then returns
  → **2 rows** (domain-exit closes the spell).
- `test_carrier_handoff_midspell_stays_one_row` — a **real** mid-spell carrier change (the ball-nearest team-A
  player actually differs across frames, verified by asserting `infer_ball_carrier` picks two different ids)
  while possession + domain are unbroken → still **1 row** (the fixture must genuinely change the carrier, not
  be a copy of the single-spell test).
- `test_period_boundary_splits_spells` — frames spanning a period boundary never merge into one spell.
- `test_one_frame_domain_blip` (R2-M4) — a single out-of-domain frame mid-spell closes + reopens (the
  off-by-one the state-machine is most likely to get wrong).
- `test_treatment_capped_by_window_T` (R2-H3) — on a *long* continuous spell, a cross at `entry + (T−1)` s is
  `Z=1`; a cross at `entry + (T+1)` s is `Z=0` (the fixed `T` cap binds).
- `test_treatment_capped_by_possession_end` (R3-M1) — on a *short* spell (ends ≪ `T` later), a cross *after*
  the spell ends is `Z=0` (the `spell_end` clamp kills cross-phase misattribution); a cross within the spell is
  `Z=1`. An opponent cross is `Z=0`.
- `test_outcome_strictly_post_cross` (R2-M1) — a shot **before** the cross does **not** set `Y=1`; a shot in
  `(t_cross, t_cross + W]` does. For a control (no cross), a shot in `(entry, entry + W]` sets `Y=1`.
- `test_score_differential_populated` (M1) — `score_differential` is non-NaN when `actions` carry results
  (patch a scoreline; assert the column reflects it), NaN-tolerant when they don't.
- `test_confounder_set_is_seven_no_ball_features` (M3) — the returned confounder columns are exactly the 7
  paper confounders + 6 GK; `ball_r/theta/speed` are absent from the causal `X` set.
- `test_carrier_coverage_reported` — per-row `carrier_resolved` is emitted; unresolved-carrier rows flagged.

### 5.3 `tests/causal/test_causal_e2e.py` (`@pytest.mark.e2e`) — exercises the integration seam (H2/R2-H2)

The e2e **must chain `build_opportunities` → `analyze` → write on real tracking frames** (reusing the
geometry-correct spell-fixture builders from `tests/causal/_fixtures.py` — single source, R3-L2 — *not*
synthetic numpy opportunity rows, the rev.2 plan's mistake the re-review caught). It covers the seam that unit
tests miss: the `build_opportunities`
column contract ↔ `analyze`'s `opp[PAPER_CONFOUNDERS]`/`opp[GK_BLOCK]` selection, the `_interval_label`
team-id join on real ids, the spell machine on real geometry, and `run()`'s coverage / eligible-pool /
`no_eligible_provider` branches.

- `test_build_analyze_write_chain` — `build_opportunities(fixture_frames, synthesized_actions)` →
  `analyze` → `_write`; assert `metrics.json` + `report.md` exist with **finite** ATT-without-GK / ATT-with-GK /
  placebo-band / `gk_nan_fraction` / `base_nan_fraction` / `ps_overlap_fraction` and boolean
  `gk_clears_placebo_band` + `causal_claim_supported`. Never asserts `gk_beats_placebo` (a null is valid).
- `test_run_with_monkeypatched_loader` — monkeypatch `iter_matches` to yield one fake `(frames, actions,
  home_team_id)` so **`run()` itself** executes (coverage filtering, eligible-pool concat, artifact write);
  assert a `metrics.json` with a `coverage` block is produced.
- `test_analyze_positivity_guard` — `analyze` on an all-zero-`Z` opportunity frame → `status ==
  "no_variation_in_treatment"`, **no NaN ATT** (M5).
- (Pure-numpy `matching.*` smoke stays in `test_matching.py`, not mislabeled as e2e.)

### 5.4 Determinism / hygiene

- No `np.random` global; explicit seeds. ruff + pyright clean; Examples on every public-ish function
  (the port is private but docstring-Exampled for maintainability).

---

## 6. ADR-015 + NOTICE

- **ADR-015** (`docs/superpowers/adrs/ADR-015-causal-validation-port.md`): records the private-causal-port
  pattern — pure numpy/sklearn matching estimator, no R, no new dependency; private `_causal/` until a second
  consumer (TF-19) justifies promotion; the report is a *reported* research artifact, never a ship/CI gate;
  Abadie–Imbens SEs are mandatory under matching-with-replacement. **Two named approximations (M7):** (i) the
  J=1 within-treatment-group `σ̂²(X)`; (ii) matching is on the *estimated* propensity score, so the
  fixed-matching-variable AI formula is **conservative** (Abadie–Imbens 2016, *Econometrica*) — acceptable for
  a reported artifact, named so a future production consumer knows to revisit it.
- **NOTICE:** extend the existing Cao et al. entry (lines 73-79) — the runtime surface is state-level
  (already noted); **PR-C adds the sender-level causal harness** as the paper-faithful arm, with the
  tracking-only-opportunity-detection + league/era caveat.

---

## 7. Real-data run + bundled report

PR-C bundles a `metrics.json` + short report from a **maintainer run on the pining corpus** (DGX Spark +
owner token), mirroring PR-B's training run. The code + known-truth unit tests + ADR are built and verified on
Windows first (the estimator is unit-tested against known truth, not real data); the DGX run is the closing
step that produces the bundled artifact. The bundled report lives under the PR-C feature commit (not a
standalone doc commit). Paper reference points: ATE 1.6%, ATT 5.0% (30 matches CSL, 2,225 opportunities / 692
crosses) — sign-sanity reference, not a reproduction target.

---

## 8. Shipping / commit structure (deferred to ship-time, per owner)

All PR-C code+tests+ADR are **structure-agnostic** — identical whether PR-B+PR-C ship as one combined 4.16.0
release or two (4.16.0 PR-B + 4.17.0 PR-C). Per the owner's "no commits until TF-17 done including PR-C," the
CHANGELOG/version/commit-structure decision is made at ship time once everything is built and verified. The
open fork (one combined release vs two back-to-back) is surfaced then, not now.

---

## 9. Out of scope (YAGNI)

- Public `silly_kicks/causal/` promotion (waits for TF-19, the 2nd consumer — ADR-015).
- R / `Matching`-package parity beyond the documented estimator semantics.
- Caliper / 1:k matching / kernel matching variants (paper uses 1:1, no caliper).
- Using the causal effect as a runtime xfn (the cross→shot effect lives in the harness, not as a feature;
  shot value is already covered by VAEP/xthreat/xS).
