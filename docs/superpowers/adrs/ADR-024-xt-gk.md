# ADR-024: xT-GK (Eyestone) — pure parametric GK-distribution-value feature

**Status:** Accepted (2026-06-08; amended 2026-06-09 — goal-kick coverage + SkillCorner completion/variant family, both folded into 4.21.0; amended 2026-06-10 — per-type base-rate serve switch, SK-91, 4.21.4; amended 2026-06-27 — PEV/DZV fidelity fix, Eyestone Q1–Q3, 4.35.0, PR-S100)
**Deciders:** Karsten (with Claude); collaborator Jeffrey Eyestone (metric author)
**Related:** ADR-005 (tracking feature surfaces), ADR-019 (id-dtype contract), ADR-020 (frame-aware xfns frame-id resolution), ADR-021 (pluggable xT), ADR-011 (trained-model lifecycle — explicitly NOT applicable here)

## Context

Jeffrey Eyestone's **xT-GK** (Expected Threat for Goalkeepers; winner, Pitch to the Pros 1, May 2025) re-values goalkeeper distribution actions (goal-kicks, keeper passes/throws), correcting traditional (Karun-Singh) xT which systematically under-values GK zones, ignores pressure, and penalises back-passes. Jeffrey gave explicit public-with-attribution permission (email 2026-06-06). silly-kicks *computes* the metric; dashboards live in the lakehouse (the architecture split).

## Decision

Implement xT-GK as a **pure parametric compute feature** — an analytical formula with no learned weights. Its architectural siblings are OBSO / DAS / pitch-control / cover-shadows (frame-aware tracking features with `compute_*` / `add_*` / `*_xfns` + atomic mirror), **not** trained-model features. **ADR-011 (code-PR → weights-PR, booster-JSON, `[xgboost]` gate, HPO trainer) does NOT apply.**

Key sub-decisions:

1. **Tracking-required.** Of the six components, five are events-derivable but **PEV (pressure-escape) needs a pressure signal, and no pressure survives SPADL conversion** (StatsBomb `under_pressure` dropped; "Pressure" events excluded). Shipping a default missing pressure-escape would be a silent-drop trap. xT-GK lives in `tracking/` and is frame-aware. Matches Jeffrey's "continuous pressure" preference.

2. **RAV's P(success) reuses the tracking xC model `get_xc`** (wraps `accessible_space.get_expected_pass_completion`) → xT-GK gates on the `[das]` extra, **fail-loud** (`ImportError("xT-GK requires the [das] extra: ...")`) — never a silent RAV-less composite. An empirical zone-grid completion model is far too sparse for the rare GK-distribution corpus.

3. **Required injected pre-fitted `ExpectedThreat` (no self-fit → no leakage).** `ExpectedThreat` is fit-only (4.17.0). `compute_xt_gk`/`add_xt_gk` take a **required `xt=`** kwarg fitted by the caller on a corpus disjoint from the scored matches (the OBSO/frozen pattern). A guard raises if `xt.xT` is all-zero (unfitted). `xt_gk_xfns` is a factory closing over the injected `xt`.

4. **Normative params are NEVER calibrated** (category error — they encode the value claim, not a predictor): γ/δ/φ/η + `v_def` + `defensive_third_boundary` + `pressure_scale` are intent-set. The lone smoothing nuisance `convolution_sigma` is hand-set (see §scan). There is **no calibration phase** and no TF-24 wiring (option (c)).

5. **Formula — Option B (Jeffrey confirmed 2026-06-08).** The destination value `xT★(z')` enters the composite's main value path **once**, owned by RAV; the base term is **origin-only** (`−xT★(z)`). A separate `progress = xT★(z')−xT★(z)` feeds PEV only (folding PEV onto the origin-only base would zero it). Composite: `xT-GK = T·(−xT★(z) + γ·PEV + RAV) + φ·DZV`. This closes the earlier `(1+p)·xT★(z')` double-count.

6. **`home_team_id` is parity-only.** xT-GK consumes **LTR-normalized** SPADL action coordinates, so `home_team_id` (required keyword on `add_xt_gk`/`xt_gk_xfns`) is accepted for GK-feature-family signature parity and CI-gate construction, and is intentionally unused by the math — documented so a future reader does not "fix" the unused arg.

7. **xthreat coupling.** `_grid_value` reuses `silly_kicks.xthreat._grid._get_cell_indexes` (a private cross-package symbol) rather than re-deriving the cell-index convention — this is xthreat's port. Pinned by `test_grid_value_pinned_to_expected_threat_rate` (asserts `_progress` at σ=0 equals `ExpectedThreat.rate`). **Promote to a public xthreat API** (e.g. `ExpectedThreat.grid_value`) if xthreat ever refactors its grid orientation.

8. **Unlinked in-scope rows are surfaced, not swallowed** — a NaN composite plus a fixed-message `warnings.warn` (count omitted so warning-dedup collapses it in the xfns hot loop) plus a NaN `link_quality_score` provenance column. Never a silent substitution.

## convolution_sigma scan

`convolution_sigma` default is **0.8** (provisional, hand-set). The one-off per-σ `xt_gk_base` distribution-stability scan on real tracking data is **owner-gated** (the three tracking providers have known data-quality issues, spec §7); it is recorded here when run. Until then, 0.8 is a defensible smoothing default; σ=0 disables convolution (`xT★ ≡ xT`).

## OOD risk (RAV / get_xc on goal-kicks)

accessible-space's xC is validated on open-play passes; goal-kicks are long aerials (a different regime). The synthetic construct gates are provider-independent and cannot detect this. An **owner-gated `@pytest.mark.e2e` smoke** (`tests/tracking/test_xt_gk_e2e.py`, set `XT_GK_E2E_MATCH_DIR`) asserts goal-kick xC is finite + non-degenerate. **Escalation (binding):** if goal-kick xC is garbage, a geometry-based completion prior for the aerial regime becomes Phase-1 scope, not a follow-up. **Outcome: ESCALATED** — the owner run measured open-play get_xc resolving for only ~31% of real goal-kicks. Resolved per the §Amendment below (a fitted `GkCompletionModel` replaced get_xc; the e2e is now a coverage/provenance smoke).

## Consent trail (audit-grade)

- **2026-06-06 email** (Jeffrey): *"Being public with attribution is OK."* + *"pick the easy answer"* / *"just do whatever is easy"* (component set = all; presets = ship; pressure = continuous; P(success) = library machinery).
- **2026-06-08 reply** (Jeffrey, to the two-option question on the `(1+p)` destination weighting + preset publication): *"1 B, 2 OK to go with provisional values."* → **Option B** (destination counted once; base origin-only; RAV sole owner of `xT★(z')`) **and** provisional in-range preset values approved for publication.
- **Scope of what's final vs provisional:** the composite `xt_gk` **form** is final (Q1 resolved). The preset **point-values** remain provisional pending an exact Q2 table; the `xt_gk` magnitudes may shift if Jeffrey later supplies exact numbers. The decomposed components are the stable deliverable.

## Consequences

- Additive: xt_gk is in **no** default xfn list (opting it into VAEP is a self-triggered retrain). No retrain trigger on existing features.
- Phase 2 (opt-in team/dataset γ/δ/φ/η empirical estimation) is a separate, later release, gated on Jeffrey's preferred estimation recipe (his Q7).
- New `[das]` dependency edge for xT-GK consumers (lazy; bare `import silly_kicks` stays light).

## Amendment (4.21.0): goal-kick coverage — RAV completion model + coordinate derivation

The owner OOD run (above) escalated: open-play `get_xc` resolved for only ~31% of real
goal-kicks, and ~67% of real GS goal-kicks carry a NaN origin. Both gaps are closed **inside
xT-GK, honestly tagged** — superseding decision §2 (RAV reuses `get_xc` / hard `[das]`):

1. **RAV's P(success) is now a fitted `GkCompletionModel`** (`_gk_completion.py`) — a **logistic**
   GK-distribution pass-completion model (sklearn at fit; pure-numpy `sigmoid(Xβ)` at serve, so
   **no new runtime dependency**), label = observed SPADL `result_id == success`. Bundled GS
   `default` (30 WC2022 matches) ships with the package; pickle-free JSON + SHA256 envelope;
   `from_variant("default")` + a caller `completion=` override on `compute_xt_gk`/`add_xt_gk`.
   **`[das]` is no longer required.** ADR-011's full trained-model lifecycle (booster-JSON,
   `[xgboost]` gate, HPO) still does NOT apply — this is a trained-*light* artifact (few
   coefficients, observable label, pure-numpy serve), the same tier as the xS/xCross precedent
   for serialization/leakage discipline but without the heavy infra.
   - **Green gate (the sole correctness gate): native-origin POOLED out-of-fold calibration.**
     `scripts/train_gk_completion.py` scores all native-origin rows out-of-fold and requires a
     bootstrap AUC CI **lower bound > 0.5** with `n_native ≥ 100` (sample-size-aware, not a
     per-fold mean over the native minority). Measured on the GS default: **AUC 0.838,
     CI95 [0.809, 0.865], n_native 1395, Brier 0.122 < base-rate 0.171**, density finite-rate
     **96%**, label split 401 fail / 1265 success (success-rate 0.76, non-degenerate).
     `prepare_gk_completion_training_data` **fails loud** on a degenerate label distribution.
   - **Missing-value policy:** per-feature density NaN → training-mean impute (neutral after
     standardization, via `nanmean`/`nanstd`); whole-row geometry-unscoreable → per-type base
     rate. The base-rate fallback is reachable **only** through the standalone
     `compute_gk_completion` — the RAV path leaves an unresolvable-destination goal-kick NaN
     (no z' ⇒ no RAV/`xT★(z')`), so the coverage claim is "~100% of goal-kicks *with a
     resolvable destination*", not all goal-kicks.
   - **`is_throw_in` is near-inert on the GS default** (goalkeepers essentially never take
     throw-ins → ~0 coefficient); kept for forward-compat, not a bug.

2. **Coordinate derivation** (`resolve_gk_geometry`, `_gk_geometry.py`): a **scoped, conditional**
   origin (native → in-area tracking-GK clamped to `x ≤ 16.5 m` → goal-area rule point
   `(5.5, 34)`) + destination (native → in-period next-event start, `(game_id, period_id)`
   boundary-guarded). The derived coordinates **feed the valuation internally and NEVER mutate
   the shared `actions` frame** — a converter-/`spadl`-level coordinate change would be a Hyrum
   + retrain trigger for every downstream consumer (VAEP, the xT fit, all features) and is the
   **D-A1** general coordinate-enrichment follow-up (its own ADR + a coordinated retrain;
   logged in TODO). The helper is built pure / provenance-emitting / xT-GK-decoupled precisely
   so it can be lifted out later. Imputed-origin rows are **usable but lower-confidence** (the
   imputed origin propagates through the model's length/forwardness features into RAV) and ship
   **tagged + gated separately**, never silently equal to native-origin values.

3. **Provenance for outside inspection.** Per-row enum columns ride in the output wide table —
   `xt_gk_origin_source` (native / tracking_gk / goalkick_prior / unresolved), `xt_gk_dest_source`
   (native / next_event / unresolved), `xt_gk_origin_confidence` (continuous) — the established
   `is_goalkeeper_source` idiom (queryable/filterable downstream). An optional aggregate
   `XtGkReport` (counts per source) mirrors `ConversionReport`/`LinkReport` for pipeline QA;
   by construction its counts equal the columns' `value_counts`.

4. **Train==serve parity** is enforced in code at **every** producer, not just by tests: one
   shared domain predicate (`_gk_distribution_mask`), `resolve_gk_geometry` resolved on the
   **full** action list *then* masked (so a NaN-destination kick's `next_event` borrows the next
   *actual* action, not the next in-domain row — the positional-`shift(-1)` blind spot), one
   shared `_gk_completion_density` producer, and one shared `extract_gk_completion_features`.
   Parity tests back each axis.

**Surfaces:** `resolve_gk_geometry`, `GkCompletionModel`, `compute_gk_completion`,
`add_gk_completion`, `XtGkReport` exported from `silly_kicks.tracking` (+ the bundled
`_gk_completion_weights/default/`). `add_gk_completion` is the **lakehouse wide-table
aggregator** -- it emits a `gk_completion` column per in-scope GK distribution (NaN
out-of-scope) for the single-pass action-context materialization, reusing RAV's exact
`_completion_p` scoring path (geometry resolved on the full action list, then masked) so the
standalone column equals the P(success) RAV consumes. It satisfies the standard `add_*`
contracts: `@nan_safe_enrichment` (ADR-003), the id-dtype invariance gate (ADR-019, registered
in the gate's `AGGREGATORS`), idempotent linkage-provenance merge, and the `links=` pre-linking
kwarg. C4 action-coupled-aggregator count **26 → 27**. (No `gk_completion_xfns`: completion is
RAV machinery, not a VAEP feature. Atomic mirror deferred -- the lakehouse path is standard
SPADL.)

## Amendment (4.21.0): SkillCorner completion — native-`result_id` fix + provider-aware variant family

Folded into 4.21.0 (single release) so SkillCorner `xt_gk` is construct-correct and poolable with
Gradient Sports. Spec: `docs/superpowers/specs/2026-06-09-xt-gk-multiprovider-completion-design.md`.

**Decision.**
- **D-S8 — fix SkillCorner `result_id` to the native pass outcome** (`pass_outcome` → `received==True`
  success-only → residual `same_team_next`), emitting a dedicated `result_source` tier column
  (`native`/`inferred`/`stopgap`). The prior bare `same_team_next` proxy was a converter
  correctness bug: it agreed with the native outcome only ~0.72–0.79 and overstated goal-kick
  success ~16 pp. Completion is corrected to native **where native fields exist**; residual rows keep
  a flagged `stopgap` for VAEP coverage. **VAEP-retrain trigger** accepted (the gold-standard one-
  construct fix is worth one more SkillCorner retrain; the lakehouse waits and re-materializes).
- **D-S1/D-S2 — provider-aware variant family, weights re-measured.** `GkCompletionModel` gains a
  variant registry (pure `variant_key_for_provider` + auto-selection from `frames["source_provider"]`,
  `completion=` override, fail-loud on >1 real provider, `snapshot` excluded). The prior 0.50
  non-transfer was on the WRONG (proxy) label and is void; **re-measured on the corrected native
  label, GS still does not transfer** (GK-pass AUC 0.412 < chance) → SkillCorner needs distinct
  weights. Bundled `skillcorner` variant: GK-pass **AUC 0.739, ECE 0.036**.
- **F1/G1 — train on the `native` tier only.** `inferred` (`received==True`) is positive-only and
  would bias the multiplicatively-consumed calibration; only `pass_outcome` supplies both classes.
- **Goal-kicks are a documented limitation.** SkillCorner goal-kick completion is chance from
  geometry (AUC 0.433) even on the native label; goal-kicks are **model-scored** (their `xt_gk` is
  on-scale per the comparability gate) but low-discrimination — a per-type base-rate serve switch is
  a deferred follow-up (TODO), not in this release. **(Superseded in 4.21.4 — see the Amendment
  below: SkillCorner goal-kicks now serve the calibrated base rate; GS goal-kicks stay model-scored.)**
- **D-S7/D-S9 + H1 — pooling safety.** `xt_gk` is pooled across providers in the lakehouse. Provenance
  columns `xt_gk_completion_variant` / `xt_gk_completion_source` + `XtGkReport.spans_multiple_variants`
  make a mixed-variant aggregation observable. A cross-provider comparability gate
  (`scripts/_xtgk_comparability.py`, owner-run, REPORTED not CI) found SC-vs-GS `xt_gk` **within
  tolerance** on matched distance bands → pool directly, no re-scale (G2: post-calibration the
  expected outcome is within-tolerance or escalate, never silent conforming; any re-scale would be a
  per-variant post-composite affine on `xt_gk`, evidence-gated). Contract: **do not pool `xt_gk` across
  `xt_gk_completion_variant` without a validated comparability**.

**Consequences.** SkillCorner completion correct for every `result_id` consumer (VAEP + features +
lakehouse SPADL), not just xT-GK — one construct, native where present. xT-GK reads `result_id`
uniformly; no side-channel. VAEP-retrain trigger for SkillCorner. No C4 enumeration change (a
completion *variant* is not a new aggregator; count stays 27). `[das]` remains optional.

## Amendment (4.21.4, 2026-06-10) — per-type base-rate serve switch (SK-91)

Ships the deferred per-type serve gate. `compute_xt_gk` consults a per-type decision baked into the
`GkCompletionModel` artifact and, for a gated type, overrides the geometric `P(success)` with the
**per-type calibrated base rate** (tagged `xt_gk_completion_source = "base_rate"`).

**Decision.**
- **The gate is one pure function — `serve_mode_from_lcb(lcb, n)`** (`_gk_completion.py`): serve the
  model iff the type's held-out AUC **lower-confidence-bound > 0.5** on `n ≥ _GATE_N_MIN` (50); a
  `None`/NaN LCB (degenerate/near-empty positive class) or too-small sample → `base_rate`. Serve uses
  the **conservative LCB**; *bundling* the variant uses the point estimate (different questions).
- **The gate lives in the artifact, not the call.** `GkCompletionModel` gains `_type_serve_mode` +
  `_type_gate_metrics` (per-type `{auc, lcb, n}` transparency), computed at train time from the
  held-out OOF over the model's 3-way `{goalkick, throw_in, other}` partition (`_per_type_gate_from_oof`,
  wired into both fits). Artifact `VERSION` → **1.1.0**; `load()` **fail-opens** (a pre-gate 4.21.0
  artifact has no `type_serve_mode` → all types serve `"model"` = prior behavior). `predict_proba`
  stays a pure scorer; the switch + tagging is in `compute_xt_gk`; the atomic mirror inherits it.
- **Data-driven per variant, measured owner-run.** Bundled **SkillCorner**: goal-kick → `base_rate`
  (AUC 0.433, LCB 0.277), throw-in → `base_rate` (degenerate, n≈2), GK-passes → `model` (AUC 0.737,
  LCB 0.674). Bundled **GS `default`**: goal-kick → `model` (AUC **0.836**, LCB 0.798 — GS goal-kick
  completion *is* geometry-predictable, unlike SkillCorner), throw-in → `base_rate` (degenerate, n=1),
  other → `model`. The committed mode for each is locked by a real-artifact test.

**Re-bundle reproducibility (the non-obvious part).** The gate is attached onto the **committed
coefficients** (the re-bundle loads the shipped model, sets the gate fields, re-saves — coefficients
ship **byte-unchanged**; the fresh full-data fit is only a corpus-identity *probe*). A landmine
surfaced: the bundled weights are reproducible **only with full-match frames**, but the documented
training command falls back to the script default `--tracking-limit 200`. At 200 frames (~20 s) the
SkillCorner *derived-GK* over-flags 3–4 goalkeepers in 2 of 10 matches (too little data), inflating
the frame-derived GK-pass domain 461→538; with full frames it is robust (2 GKs/match) and the corpus
reproduces **exactly** (event-pinned goal-kicks were stable throughout — the tell). GS reproduces its
row set exactly at full frames but its coefficients differ by ≤0.0056 — an **unrecorded original
`tracking_limit`** loaded a partial frame subset (density-finite 96.3% vs 98% at all-frames), an
irreducible float difference. Decision: the corpus-identity guard moved from byte-identity (`atol=1e-9`)
to a **meaningful tolerance** (`_CORPUS_IDENTITY_ATOL = 0.05`) — it still aborts on a real retrain
(the earlier SkillCorner data-drift retrain shifted coefficients ~0.47, ~9× the floor) while
tolerating tracking_limit density noise; the served coefficients are byte-identical regardless, since
the re-bundle ships the loaded committed model. **The `train_gk_completion.py` `--tracking-limit`
default was changed `200` → `None` (full match)** so the bare documented command reproduces — the
generic 200 was a per-model footgun for this model (a small frame cap starves the SkillCorner
derived-GK and collapses the GS density feature); both model cards record it.

**Consequences.** Not a VAEP retrain (xt_gk is opt-in, in no default xfn list) — but an `xt_gk`
serve-output change for the flipped types: the lakehouse re-materializes `xt_gk` for **SkillCorner
goal-kicks (~15% of its GK-distribution rows) + degenerate throw-ins (both variants)**; GS goal-kicks
unchanged. No C4 enumeration change (a serve gate on an existing model is not a new aggregator/model/
backend; count stays 27). `compute_gk_completion` (standalone) is unaffected — it already base-rates
geometry-unscoreable rows; the gate governs only the in-scope, geometry-resolved RAV path.

## Amendment (4.35.0, 2026-06-27) — PEV/DZV fidelity fix (Eyestone Q1–Q3)

Eyestone reviewed the shipped 4.21.x formulation and answered three open fidelity questions
(2026-06-27). Two terms were re-derived to match the published framework; **Option B (§5) and RAV
are unchanged**.

**CHANGE 1 — PEV reads the GK-revalued surface (Q1 + Q2 are one fix).** The pressure-gated rectified
form is kept exactly: `PEV = ρ·max(0, progress)`. What changed is the surface the forward gain is
measured on: `progress = V_GK*(z′) − V_GK*(z)` on the **revalued** surface `V_GK = xT ⊙ φ(z,d)`
(convolved with `convolution_sigma` like `xT*`), **not** raw `xT*`. On the raw grid the keeper-zone
forward gain is ~0 — the measured PEV inertia — because keepers sit in the flat part of the xT
surface; revaluing the surface is the whole point (it is *not* tournament-dependent). PEV stays a
pressure-gated forward **gain** (not a destination level), so RAV remains the sole owner of the
destination and Option B is untouched — no double-count.

**CHANGE 2 — DZV is the published revaluation multiplier (Q3), scale-reconciled via Option A.** The
old additive `v_def − xT_raw(z)` back-pass floor is replaced by the deck form
`M(z) = φ(z,d)·[1 − V_GK(z)/max V_GK]`. A literal `M` is ~2.5/action and would swamp the O(0.01)
base/RAV/PEV terms (Eyestone's explicit scale anchor: La Liga DZV ≈ +0.27/match ≈ 0.009/action).
Per his "the multiplier must revalue a small possession base (not be added raw)," DZV is the
revaluation **increment** on the origin possession value, `(M−1)·V_GK(z)`, gated to the defensive
third — measured O(0.01) on the unit fixtures (~0.006–0.018 for realistic deep `V_GK` 0.005–0.01).
**Option A (increment) over B (revalued total) / C (revalue a fixed baseline)** because the
increment is the value the revaluation *adds*, keeping it orthogonal to `base` (which already
surrenders the origin's raw threat) — B would re-credit the full revalued origin and partially undo
Option B's clean "origin surrendered" semantics.

**φ(z,d)** `= α·(1 − d/D_max)^(−β)` for `d < D_threshold`, else `1`; `d` = LTR origin x. `α = 2.1`,
`β = 0.8` are **canonical**; `D_max = 105`, `D_threshold = 35` (= `defensive_third_boundary`) are
provisional (labelled like the γ/δ/η presets). `XtGkParams` gains `dzv_alpha` / `dzv_beta` /
`dzv_d_max`; the now-dead `v_def` is retired. The scalar `phi` param stays the preset-modulated
overall DZV weight (canonical *shape* lives in the φ grid).

**Invariant (Eyestone's explicit constraint).** φ enters value through **PEV and DZV only** — `base`
keeps `−xT*(origin)` and RAV keeps `xT*(z′)` / `xT*_counter` on the raw `xT*` surface, so the
destination is never revalued twice and Option B holds. Guarded behaviorally:
`test_phi_shape_changes_only_pev_and_dzv_not_base_or_rav` (changing α/β leaves `xt_gk_base` +
`xt_gk_rav` byte-identical) and `test_pev_reads_revalued_surface_not_raw` (a deep build-out with ~0
raw gain produces PEV only with revaluation on). The four-term sum identity
`T·(base + γ·pev + rav) + φ·dzv == xt_gk` is pinned by `test_composite_discounts_threat_terms_only_not_dzv`.

**Post-run verification (Eyestone's "confirm post-run" ask — for the lakehouse re-run report).** Beyond
the ~0.01 mean: (1) DZV per-action **distribution + by-zone profile** (does it peak toward the top of
the defensive third, per M(z)?); (2) PEV lights up for the **right** actions — short deep build-out →
positive PEV; long clearances out of the third may stay ≈0 (origin revalued above destination →
`max(0,·)=0`); (3) sanity-check the `d=35` φ discontinuity (2.9→1) creates no boundary artifact for
distributions landing right at the defensive-third line. Forward note (NOT this change): computing PEV
on `V_GK` is the first half of Eyestone's receiver-pressure extension (adding a receiver-pressure term
`q`) — future work.

**Consequences.** Not a forced VAEP retrain (xt_gk is opt-in, in no default xfn list) — but an
`xt_gk` serve-output change: the lakehouse re-materializes `fct_action_context` and re-runs the
WC2022 cohort/report. No C4 enumeration change (no new aggregator/model/backend). Atomic mirror
inherits.

## References

- Eyestone, J. (2025). *Expected Threat for Goalkeepers (xT-GK)*. Pitch to the Pros 1.
- Singh, K. (2018). *Introducing Expected Threat (xT)*.
- See NOTICE for full bibliographic citations.
