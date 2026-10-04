# Research batch 2026-10 — vetting record for TODO rows TF-68 … TF-73 and the TF-50 amendment

**Status:** revision 8 (2026-10-03). It applies the owner's rulings on the seven batch decisions (see "Owner decisions — RULED"). **Reviewer B approved revision 6** (`D:\Development\_reviews\2026-10-03-todo-research-batch-B-r6.md`, APPROVE, 3/3 passes, no open finding); revision 7 adopted only B-r6's optional wording tightening. The ruling-driven text in revision 8 post-dates that review and has not been independently reviewed. Revisions 2–6 applied rounds 1–5 (`…-B.md` through `…-B-r5.md`). Findings are verified in the tree before they are applied or surfaced, including claims that originate with the reviewer. One revision-3 claim was marked verified but was not checked; see R3-W below.

**What this is:** the evidence behind TODO rows added on the owner's instruction (2026-10-03) to "add properly vetted list items with new tasks that make sense". It is not a spec. Each row still needs its own brainstorm → spec → plan → review cycle.

**Where claims were checked:**
- Code claims: `main@ba9c151`.
- The G2 corpus-policy test, not yet on `main`: the combined-cycle C1 amend `0cdcf05`.

This page quotes no restricted numbers and names no private data. TODO line numbers shift, so rows are cited by ID.

**The bar.** Settled owner decisions (2026-10-03):
1. New frameworks are acceptable as optional extras.
2. No lakehouse work; TF-70 lives in silly-kicks.
3. Restricted-data artifacts may ship if non-reversible (ADR-038 is a labelling control, not a public-only rule).
4. No optical tracking is expected soon.
5. TF-70: exact latent-Gaussian estimator, team strength estimated jointly.
6. TF-50 metabolic-power amendment applied.
7. Learned EPV retired; TF-73 = a bundled xT.
8. The seven batch decisions, ruled 2026-10-03 (see "Owner decisions — RULED").

Rows were screened against:
- the **raw-primitives convention** (`AGENTS.md`; `docs/context/conventions-core.md:66`; recorded as ADR-110 on 2026-10-04 — it was cited as ADR-009, the precedent TF-45 modelled it on; see the end);
- the injected-xG boundary (`docs/context/xt-gk.md:17`, `ADR-097:14`);
- E1–E7 and the validity ladder (`docs/research/README.md`).

## Sources

**Read in full:**
- Andorra & Göbel 2024 (SFM, arXiv:2412.05911);
- Egidi & Gabry 2018 (JQAS 14(3):143–157);
- Van Roy et al. 2021 (arXiv:2104.03252 v2);
- Fernández, Bornn & Cervone 2020 (arXiv:2011.09426);
- Schiettecatte & Van Haaren 2026 (B-CLAD, MLSA 2026 paper 380);
- Bekkers 2025 (arXiv:2501.04712 v2);
- Sá-Freire et al. 2026 (VAEP-360, MLSA 2026 paper 277).

**Abstract-level screen:** arXiv 2311.13707, 2511.23072, 2508.05891, 2205.07193, 2507.10626, 2603.15212, 2305.17886, 2411.17450.

**Methodology references in the rows:**
- Abowd, Kramarz & Margolis 1999; Andrews, Gill, Schank & Upward 2008; Kline, Saggio & Sølvsten 2020 (two-way effects, limited-mobility bias);
- Zhao, Small & Bhattacharya 2019 (marginal sensitivity model); Abadie & Imbens 2008 (failure of the bootstrap for matching estimators); Otsu & Rai 2017 (weighted bootstrap for matching estimators);
- Minetti et al. 2002 (energy cost of running on slopes).

## Revision log

### Owner rulings (2026-10-03) → revision 8

The seven decisions were ruled after Reviewer B's APPROVE. Each ruling is applied to its row; the options analysis below is kept as the record of what was weighed.

| # | Ruling | Applied to |
|---|---|---|
| 1 | TF-68: (A) probability models only; the ghost regressors are out of this row | TF-68 title, population, size (Wicked, slot no longer provisional) |
| 2 | TF-71: E-value **and** the marginal sensitivity model | TF-71 title, scope, size (Wicked); moved from the Dunkin' band to the head of the new Wicked rows (least effort of the four) |
| 3 | TF-73: route A (default flip) plus a named opt-in | TF-73 delivery; the pre-existing `xt=` gap is moot |
| 4 | TF-73: (i′), refined — one ball-centred (zero-mean) kernel for all three paths, σ fitted on the public StatsBomb open data with a pre-registered estimator, re-centred on each frame's ball; a direction-dependent (ii) only if the OBSO predictive-validity gate fails | TF-73 transition, gates, size (Wicked, slot no longer provisional) |
| 5 | TF-73: (a) `scoring_prob_matrix` is an internal xT component; the bundle is an `ExpectedThreat`; guard: it never becomes a per-shot xG source | TF-73 artifact, gates |
| 6 | TF-73: bundle under `silly_kicks/xthreat/`; widening G2 to derive its population (ADR-056) is a prerequisite | TF-73 placement, gates, sequencing; the G2 note under "Outside this batch" |
| 7 | Ordering: keep the strict rule; TF-73 stays last in the Wicked band | TF-73 size note |

**Author-found correction (while applying ruling 2):** revisions 2–7 let TF-71's spec choose "between the delta method … and a bootstrap" for the E-value's bound. The naive bootstrap is invalid for matching with replacement (Abadie & Imbens 2008), and `causal/matching.py:5-6` already says so. TF-71 now requires matching-valid inference for both analyses (the delta method with a matching-valid variance; a matching-valid resampling scheme such as Otsu & Rai 2017 for the sensitivity interval). TF-71 also gains gates: the E-value reproduces the published worked examples; the sensitivity range collapses to the unconfounded estimate at Γ = 1; a planted confounder is covered at its true Γ.

### Round 6 (B-r6, APPROVE) → revision 7

| # | Item | Verified | Disposition |
|---|---|---|---|
| Observation (not a finding) | "can change which frame attains `peak`, and therefore `pausa_temporal`" is causally imprecise: under a per-frame anchor the ratio (ppcf_e·K_e) / (ppcf_t*·K_t*) moves even when the winning frame is unchanged | yes (`_obso.py:436-447`: with the factor at the target varying per frame, it no longer cancels between `actual` and `peak`) | Adopted B's optional wording in the three live sentences (TODO row, per-row (i′) bullet, decision 4 (i′)) and in the R4-1 log cell: "…so it no longer cancels in actual / peak: `pausa_temporal` changes, and so can the frame that attains `peak`." |

### Round 5 (B-r5) → revision 6

| # | Finding | Verified | Disposition |
|---|---|---|---|
| R5-1 | Decision 4 (i) said `peak` and `pausa_temporal` are "unaffected"; under (i) `actual` (`_obso.py:438`) and `peak` (`:447`) both lose the mid-pitch Gaussian's value at the target (max 1 at the centre, `:89-100`), so their values scale by 1/G per pass (about 2.4× near the penalty spot, about 16× at a corner). Only the ratio and the winning frame are invariant. As written, (i) read as value-neutral beside a "value-changing" (i′) | yes (code as cited; G = exp(−d² / 0.18) on the unit pitch gives ≈0.42 at the penalty spot and ≈0.062 at a corner) | Decision 4 (i) now matches the row: `actual` and `peak` scale by a common per-pass constant, so (i) changes `obso_actual` / `obso_peak`; `pausa_temporal` and the frame that attains `peak` are unchanged; the shape change is confined to `optimal` / `pausa_spatial`. (i) is labelled "value-changing" in the row and in decision 4. The per-frame-anchor sentences (row, per-row section, decision 4) now say the anchor can change which frame attains `peak`, and therefore `pausa_temporal`. Decision 7's parenthetical covers any per-frame-anchored ball term, (i′) or a ball-relative (ii). The R4-1 log cell no longer restates the wrong invariance. |

### Round 4 (B-r4) → revision 5

| # | Finding | Verified | Disposition |
|---|---|---|---|
| R4-1 | Option (i)'s stated pass-path consequence was wrong: `peak` scans frames at a fixed target, so static grids cancel in `pausa_temporal`; "ignores distance to the ball" is the status quo on `main`, not a cost of (i) (wording from B-r3, transcribed in revision 4) | yes (`trans_at_target` / `epv_at_target` computed once at `_obso.py:436-437`, reused in the frame loop `:447`; teammate scan `:475-477`; `pausa_temporal` / `pausa_spatial` at `_pausa.py:110` / `:113`; mid-pitch Gaussian `:89-100`) | Row, per-row section and decision 4 now state: any static transition or EPV grid is a common per-pass factor of `actual` and `peak`, so their values scale together while `pausa_temporal` and the frame that attains `peak` are invariant (where the factor at the target is non-zero); (i)'s shape change is confined to `optimal` / `pausa_spatial` (the mid-pitch weighting is dropped); the pass path is ball-agnostic on `main` and under (i); only (i′) or a ball-relative (ii) adds a ball term. New, author-added: a per-frame (i′) anchor no longer cancels in actual / peak, so `pausa_temporal` changes (spec-time choice; wording tightened in revision 7). Decision 4's wording of this was partial in revision 5; corrected in revision 6, see R5-1. |
| R4-2 | Decision 3 × decision 4 interaction unstated: route B is "non-breaking" only for some transition options | yes (`compute_pass_obso`'s `transition_grid` parameter, `_obso.py:295`, serves default and explicit callers; split note `features.py:6009-6015`) | Stated in the row ("Interactions"), in the per-row section, and in decisions 3 and 4: route B is non-breaking only with (i) or a (i′) kernel gated to the opt-in; (ii)'s reconciliation changes values for asymmetric-`transition_grid=` callers under either route. |
| NITs | TF-73 Size cell flat; decision 7 "every default number"; "all three factors AT THE TARGET" | yes | Size cell "Wicked ((i) / (i′)) / toward Monstah ((ii))", slot provisional pending the transition decision. Decision 7 excepts `pausa_temporal`. Target wording narrowed to `actual` / `peak`; `optimal` reads at teammate positions. |

### Round 3 (B-r3) → revision 4

| # | Finding | Verified | Disposition |
|---|---|---|---|
| R3-1 MAJOR | Transition option (i) "all-ones, keeping the ball-anchored kernel" is wrong on the main OBSO path (wording originated in B-r2) | yes (`compute_pass_obso` uses `ppcf × trans_at_target × epv_at_target`, no ball term: `_obso.py:436-438`; reached by `add_obso` / `obso_xfns` / single xfns via `_precompute_obso_lookup`, `features.py:5961-5977`; the ball kernel lives only in `compute_obso_surface` `:261-268` and `compute_space_created` `_space_creation.py:271-276`) | TF-73 now states each path's behaviour. Option (i) was re-described: on the pass path the transition drops out and peak/optimal ignore distance to the ball (the `peak` half and the attribution were wrong; corrected in revision 5, see R4-1). **New option (i′):** all-ones plus a ball kernel in `compute_pass_obso` (value-changing). "Consistent with ADR-041" is narrowed to the orientation-neutral half; ADR-041's "ball-anchored" premise does not hold on the pass path today. Each option is sized: (i) small, (i′) moderate, (ii) large / toward Monstah. |
| R3-W (wrong claim) | "some bundles record `training_commit`, while G2 checks `run_commit`" (originated in B-r2) | yes (G2 at `0cdcf05` checks corpus-policy keys and `run_tree_dirty` only, `:116-121`; no commit field) | **Corrected.** Revision 3 marked this "verified: yes" without checking G2 for a commit field — an author error. The widening cost is restated in decision 6: per-bundle policy-field heterogeneity plus the flat `weights/` layout. |
| R3-2 | The TF-73 artifact and route-B mechanism pre-implemented decision 5(a) | yes (`ExpectedThreat.save` → `scoring_prob_matrix`, `_model.py:360/412`; `str` slot `_physical.py:34`) | The artifact, route-B entry point and route-A token are conditional on decision 5; the interaction between decisions 3 and 5 is stated. |
| R3-3 | Route-A leaks (the decision-7 value argument; the indirect EPV trigger); the trigger's floor was missing from TF-73's gates | yes | Both are now route-conditional, and an **OBSO predictive-validity gate** was added to TF-73's gates, with its corpus and power (public tracking: IDSSE 7 + SkillCorner 20, E7) a spec-time check. |
| NITs | TF-71 size and slot; TF-68 slot; contradictory gap wording; title leans A | yes | TF-71 Size "Dunkin' (E-value only) / Wicked (+ marginal sensitivity model)". TF-68 Size "Wicked (A) / upper Wicked (B)". Both slots are marked provisional. Gap: "moot under route A; must be closed under route B". Title: "OBSO-family EPV / transition surfaces". |

### Round 2 (B-r2) → revision 3

| # | Finding | Verified | Disposition |
|---|---|---|---|
| R2-1 MAJOR | TF-73 was written as route A while the route was open; route B under-specified (passing `xt=` already silences the warning on a half-synthetic surface) | yes (the warning fires only on `epv_source == "synthetic"`: `tracking/features.py:6189/6291/6518/6677/6852`) | Row retitled neutrally. Scope and consequences are now conditional on the route. Route B requires a transition opt-in **and** closing the pre-existing `xt=` gap. Both routes sized. |
| R2-2 MAJOR | "Replace BOTH defaults" collided with a documented, deferred orientation split; ADR-041's rationale not engaged; `transition_grid` is pitch-absolute | yes (`features.py:6007-6016`; `_space_creation.py:262-270`; ADR-041 `:166-170`; ball kernel `_obso.py:261-268`) | Transition is now an owner decision: (i) neutral all-ones static grid (symmetric, no split, consistent with ADR-041) or (ii) fitted transition (the split reconciliation as a value-changing prerequisite, plus an API change). ADR-041's rationale is cited and engaged. The "down-weight" wording is corrected (the EPV ramp is flat in y; the bias is in the static transition grid). |
| R2-3 | TF-68 presumed decision 1 | yes | Population shown as options (A) and (B), with the title and size conditional. Size revised to Wicked under (A). |
| R2-4 | TF-71 presumed decision 4 | yes | Scope options listed: inside (Wicked), separate row, or not pursued. Row reads "v1 = E-value, scope pending". |
| R2-5 | Decision 3 offered one option | yes | Both options with costs: (a) rule `scoring_prob_matrix` internal; (b) bundle a values-only grid via `epv_grid=`. |
| R2-6 | TF-70 identification: team effects are per season and skill is a random walk, so same-season links are the hard ones | yes (reasoning) | The census counts same-season multi-team players separately from between-season movers; spec adds a random-walk-variance sensitivity check. Club + national team is a same-season link, kept. |
| R2-7 | The ordering label contradicted the rule | yes (`TODO.md` On Deck header) | Ordered strictly by the rule (TF-71 in Dunkin'; Wicked TF-68 → TF-69 → TF-73; Monstah TF-70). The value argument for TF-73 moved here (decision 7). |
| R1-18 (partial) | Indirect EPV trigger was a placeholder | — | Named below. |
| R1-4 (gap) | Provenance-field heterogeneity | **no — see R3-W** | Added to the G2 widening note (decision 6) as "G2 checks `run_commit`", which was wrong; corrected in revision 4. |
| NITs | `build_spells`; `MarkovPossessionValue`; `strength_column`; σ line; CI `.[test]` job; xSuccess bias loader | yes | All corrected (see the row evidence). |

### Round 1 (B) → revision 2 (summary)

- All 20 findings were verified and applied or routed to owner decisions:
  - TF-68 population (+ `GkRetentionModel`); TF-73 transition grid, entry points, recipe, G2 scope, population and model card;
  - TF-69 calibration gate, shrinkage, prior art; TF-70 AKM identification, precondition census, pre-registered gates, two-leg oracle, model details;
  - TF-71 E-value v1; TF-50 guardrails; ordering by the documented rule;
  - rejection rationales; line pointers.
- See `…-B.md` for the full list.

## Row evidence (revision 8)

### TF-71 — causal sensitivity (On Deck, Wicked; E-value + marginal sensitivity model, ruled)

`CausalEstimate`: `estimate`, `se`, `balance`, `n_focal`, `matched` (`causal/matching.py:31`). The matcher is 1:1 nearest-neighbour WITH replacement, no caliper (`causal/matching.py:3`, `:67-68`). Outcomes are binary windowed events (`causal/opportunities.py:73`). Rosenbaum Γ bounds assume disjoint matched pairs, so with reused controls the design-appropriate Γ analysis is the marginal sensitivity model (Zhao, Small & Bhattacharya 2019); the E-value alone is a generic bound that ignores the matched design. The propensity is a near-unregularised logistic model on standardised covariates (`causal/matching.py:39-64`). Inference must be matching-valid: the naive bootstrap fails for matching with replacement (Abadie & Imbens 2008; `causal/matching.py:5-6`).

### TF-68 — per-prediction feature attributions (On Deck, Wicked; population (A), ruled)

| Model | Estimator | Evidence |
|---|---|---|
| VAEP (per label) | `XGBClassifier` / `CatBoostClassifier` / `LGBMClassifier` | `vaep/learners.py:48/87/130`; `vaep/base.py:328`; value `:386` |
| xShot | xgboost; bias via `load_xgb_booster_base_score_safe` | `tracking/_xshot_occurrence.py:606`, `:715` |
| xCross | xgboost | fit at `tracking/_xcross_attempt.py:665` |
| `XSuccessModel` | xgboost (+ isotonic) or per-type LR; own bias loader `_load_booster_base_score_safe` | `xsuccess/_model.py:195`, `:219-233`, `:62` |
| `PassCompletionModel` | LR | `expected_passing/_model.py:131` |
| `GkCompletionModel` | LR | `tracking/_gk_completion.py:103` |
| `WinProbabilityModel` | LR | `win_probability/_model.py:254` |
| `ReceiverModel` | LR | `tracking/_receiver.py:326` |
| `GkRetentionModel` | LR | `xtgk/_retention.py:70` |
| `GhostGkModel` (option B; out by ruling) | HistGradientBoosting + RF-CDE | `tracking/_ghost_gk.py:1958` |
| `GhostOutfieldModel` (option B; out by ruling) | HistGradientBoosting | `tracking/_ghost_outfield.py:660` |

**CI:** `.[test]` (`ci.yml:38`) and `.[kloppy,xgboost,test]` jobs exist; nothing installs CatBoost or LightGBM.

**Size:** about 10 `contributions()` methods across 7 packages, plus a new CI job and exact-Shapley gates. That makes it Wicked, not Dunkin'. The ghost models output a position or a density, with no single scalar to attribute; attributing them would first need its own construct definition.

### TF-69 — xT policy counterfactuals (On Deck, Wicked)

- **Van Roy et al. §2–3:**
  - a per-team MDP;
  - proportional policy edits;
  - N = (I − Q)⁻¹ times possession-start counts;
  - a per-zone shot-quality adjustment;
  - an average 11.38% season-goal error (§3.2.2), against headline gains of +0.5–1.5 goals.
- **Existing parts:**
  - `XtZoneCounts` (`xthreat/_grid.py:214`);
  - `fit_from_counts` (`xthreat/_model.py:437`, no smoothing);
  - `value_iteration` (`xthreat/_value_iteration.py:13`);
  - `build_spells` (`team_metrics/_possession.py:73`; `start_x_ltr` / `last_x_ltr`, `is_open_play`).
- **Prior art:** `MarkovPossessionValue` (`xtgk/_markov.py:1-6`; `fit` requires a `pressure_column`); `xg_scoring_prob` (`xtgk/_xg_reward.py:29`).

### TF-73 — OBSO-family EPV / transition surfaces (On Deck, Wicked; route A, transition (i′) refined, `scoring_prob_matrix` (a), `xthreat/` placement — ruled)

**Ruled design (2026-10-03):**
- **EPV:** a bundled `ExpectedThreat` fitted on the public StatsBomb open data, stored under `silly_kicks/xthreat/` (the package whose `require_fitted_xt` reserves the bundled-name `str` slot, `xthreat/_physical.py:34`).
- **Transition:** the static grid becomes all-ones, and one ball-centred (zero-mean) kernel serves all three paths, with its normalisation single-sourced. Its width is fitted on the public StatsBomb open data (ball displacement to the next on-ball event) by an estimator pre-registered at spec time. It replaces today's unfitted defaults: `ObsoParams.sigma_x` / `sigma_y` = 26.25 / 17.0 m, rescaled from 120×80 constants (`tracking/_obso.py:42-44`, `:57-58`), and `compute_space_created`'s `obso_sigma_x` / `obso_sigma_y` (`_space_creation.py:97`). On the pass path the kernel is re-centred on each frame's ball, because OBSO at frame t is defined with the ball at t. Frames carry the ball as the `is_ball` row (`tracking/schema.py:29`); the spec fixes the policy for a frame with no ball position (honest NaN, never a guessed anchor). A zero-mean kernel is point-symmetric, so the deferred orientation split stays latent.
- **Delivery:** route A (default flip) plus a named opt-in through the same `str` slot.
- **xG boundary:** `scoring_prob_matrix` is an internal xT component; no public path turns it into a per-shot xG column, and the injected-`xg_column` consumers stay injected-only (pinned by a test).
- **Corpus-policy coverage:** G2 must first be widened to derive its population from every bundled-weights directory (ADR-056).
- **(ii)** (a fitted, direction-dependent transition) is pursued only if the OBSO predictive-validity gate fails.

**The three OBSO paths:**
- **Pass path:** `compute_pass_obso` has no ball term. `actual` and `peak` read pitch control × transition × EPV AT THE TARGET (`tracking/_obso.py:436-438`); `optimal` reads them at each teammate's position (`:475-477`). It is reached by `add_obso`, `obso_xfns` and the single xfns `obso_actual` / `obso_peak` / `obso_optimal` via `_precompute_obso_lookup` (`tracking/features.py:5961-5977`), and by `add_pausa` via `add_obso` (`:6777`); its actual / peak / optimal values feed PAUSA's ratios.
- **`peak` scans frames, not cells.** The transition and EPV values at the target are computed once (`:436-437`) and reused in the frame loop (`:447`). Any static grid is therefore a common per-pass factor of `actual` and `peak`, and `pausa_temporal` = actual / peak (`_pausa.py:110`) is invariant to it wherever that factor is non-zero at the target. Only `optimal`, and so `pausa_spatial` (`_pausa.py:113`), depends on the grids' shape.
- **Surface path:** `compute_obso_surface` multiplies the transition by a ball-anchored distance kernel (`:261-268`).
- **Space creation:** does the same (`_space_creation.py:271-276`).

**The two synthetic grids:**
- **Static transition:** a mid-pitch Gaussian (`tracking/_obso.py:89-100`).
- **EPV:** an x-only ramp (`:103-106`).

**ADR-041 (`:166-170`)** kept the synthetic transition as "ball-anchored and orientation-neutral". It is orientation-neutral (symmetric). It is ball-anchored only where the kernel applies (surface, space creation); on the pass path the default is the mid-pitch Gaussian alone, so the pass path is ball-agnostic on `main`.

**What each transition option does** (the record behind ruling 4):
- **(i) all-ones:** the transition drops out of the pass path. `actual` and `peak` change only by the same per-pass constant, so `pausa_temporal` is unchanged. `optimal`, and so `pausa_spatial`, lose the mid-pitch weighting in the teammate scan. The pass path stays ball-agnostic, as on `main`. The other paths keep their kernel.
- **(i′) all-ones plus a ball kernel in `compute_pass_obso`:** adds a ball term to the pass path; all three paths become ball-anchored, with no orientation split. A per-frame anchor (rather than the event frame's ball) makes the transition factor vary across the window's frames, so it no longer cancels in actual / peak: `pausa_temporal` changes, and so can the frame that attains `peak`. An event-frame anchor cancels in the ratio. The anchor is a spec-time choice.
- **(ii) a fitted transition:** asymmetric, so it activates the deferred orientation split and needs an API change. Its ball-relative forms (a displacement kernel; origin-conditioned rows from `destination_profiles(model, origin_x, origin_y)`, `xthreat/_counterfactual_seam.py:55`) also add a ball term to the pass path.

**Route × transition:** `compute_pass_obso` serves both default and explicit `transition_grid=` callers (`_obso.py:295`). Under route B, a (i′) kernel would have had to be gated to the opt-in. Under the ruled route A, the kernel applies to explicit callers too, so the release notice covers them. (ii)'s split reconciliation changes values for callers passing an asymmetric `transition_grid=` under either route.

**Warning scope:** it fires only when the EPV source is synthetic (`features.py:6189` et seq.). Passing `xt=` therefore already silences it while the transition stays synthetic. This gap predates the batch; it is moot under the ruled route A, because the default transition is no longer synthetic.

**The deferred orientation split:**
- `_precompute_obso_lookup` never flips `transition_grid`; `compute_space_created` point-reflects it (`features.py:6007-6016`; `_space_creation.py:262-270`).
- It is latent on a symmetric grid. Reconciling it changes shipped values. The ruled design keeps it latent: the static grid is all-ones, and the kernel is centred on the real ball in frame coordinates and never reflected (the rule `_space_creation.py:261-266` already states). It stays a pre-existing, documented deferral for callers who pass an asymmetric `transition_grid=`.

**Parts:**
- `scripts/_xt_corpus.py`: Singh only — "a KDE request raises", "NO smoothing/prior".
- `scripts/_sb_open_data.py`: public-only loader.
- `xthreat/_eval.py`: "NOT an xT-quality metric".
- `ExpectedThreat.save` (`xthreat/_model.py:412`); `scoring_prob_matrix` (`:360`).
- `require_fitted_xt`'s `str` slot (`xthreat/_physical.py:34`).
- `WEIGHTS_CLASSIFICATION` (`tests/test_bundled_weights_classification.py:21`).
- Kernel defaults: `ObsoParams.sigma_x` / `sigma_y` (`tracking/_obso.py:57-58`); `obso_sigma_x` / `obso_sigma_y` (`_space_creation.py:97`).

**Fallback entry points** (all covered by the route-A release notice):
- `add_obso` / `add_space_creation` / `add_pausa` and their `*_xfns`;
- `compute_obso_surface`, `compute_pass_obso`, `compute_space_created`;
- the single xfns `obso_actual` / `obso_peak` / `obso_optimal`, which carry no token;
- the changed kernel defaults, and explicit `transition_grid=` callers on the pass path.

**Population:** 80 competition-seasons / 3,961 matches (`CHANGELOG.md:111`).

### TF-70 — team-adjusted player-skill posterior (On Deck, Monstah)

- **Identification:** AKM-style through players on more than one team. Same-season multi-team membership gives hard links (team effects are per season); between-season movers link only through the random-walk prior. Limited-mobility bias applies when such players are few.
- **Precondition census** on the public corpus, splitting same-season multi-team players from between-season movers. The QA-bundle (e) row is "blocked on multi-season data", so TF-70 blocks if the census cannot power the gates.
- **Gates:** pre-registered at spec time, with "unmeasurable" as a terminal outcome.
- **Oracle:** two legs.
- **Precedent:** `duels/__init__.py:1`.

### TF-72 — MCP analytic tools (Blocked/Deferred, after TF-67 + TF-68)

`check_orientation` (`mcp/server.py:135`), `diagnose_provider`, `validate_construct_validity` (`:204`).

### TF-50 amendment — estimated metabolic power (applied)

**Sources:** di Prampero 2005, Osgnach 2010, Minetti 2002.

**Guardrails:**
- a stencil-wide detection gate;
- NaN or clip outside the validated slope range;
- a filter-sensitivity report;
- the Buchheit 2015 caveat plus the broadcast caveat;
- floodlight (MIT) as an implementation oracle only.

SkillCorner's `physical.parquet` has no metabolic field.

## Considered and not added

| Candidate | Why not a silly-kicks row |
|---|---|
| Bayes-xG-style player-adjusted xG *models* | Rejected on the no-xG-model boundary alone. TF-37(a) is narrower: pose features as xG inputs for lakehouse xG-v2, blocked on data. TF-70's optional finishing channel is a player effect relative to injected xG (effect-only output; relative to the injected xG's information set). |
| Learned frame-conditional EPV | **Retired (owner-approved).** The blocker is E7, not volume or release: across all 20 public SkillCorner broadcast matches, outfield detection is 52.5–70.1% (median 60.0%) and all ten outfielders are visible in only 3.0–22.2% of team-frames (median 9.6%) (`docs/research/gk-observability/outfield_detection_public.csv`; SkillCorner only). The achievable part (VAEP-360's context features in a GBM VAEP) matches silly-kicks' frame-aware-xfns pattern (ADR-005). **Revisit triggers:** *direct* — full-observability data at volume, or broadcast detection above a pre-set threshold. *Indirect* — **TF-73's OBSO predictive-validity gate** (now among TF-73's gates). `obso_actual` computed on TF-73's surfaces (the new default, ruling 3) must predict the possession's subsequent shot / goal outcome on held-out matches better than the synthetic baseline, above a floor pre-registered in TF-73's spec. Failing it reopens learned EPV. |
| B-CLAD | The owner's 2026-09-09 drop stands on its own. A team rating could still feed `WinProbabilityModel.fit(…, strength_column=…)` (`win_probability/_model.py:245`), but the drop stands. |
| Bayesian change-point | Generic statistics; a consumer analysis. |
| Learned tracking xPass (GNN) | A benchmark at most; torch / PyG. Watch. |
| Multi-agent trajectory imputation | E7 by definition. |
| HIGFormer, ScoutGPT | Team- or transfer-level; heavy; not primitives. |
| Multi-agent DRL off-ball valuation | Overlaps TF-35, OBSO and space creation. |
| Pressing Intensity; propensity matching | Covered (`bekkers_pi`; `silly_kicks.causal`). |
| Kinematics, sprints, attack direction; team shape; clustering, embeddings, archetypes | Planned (TF-65 / TF-50) or covered; archetypes stay with consumers. |

## Sequencing and dependencies

- **TF-58 → TF-64 → TF-66** (parked). TF-64 gates `pre_shot_gk_*`. TF-73 edits `tracking/_obso.py`, `tracking/_space_creation.py` and `tracking/features.py`: sequence or rebase.
- **TF-73 needs the widened G2 first** (ruling 6). G2 lands with the combined-cycle completion. It can be widened in that cycle or as TF-73's first step; either way, the widening lands before TF-73's bundle.
- **TF-70's oracle precedent** (the das-reference contract job) lands with the combined-cycle merge.
- **TF-72** follows TF-67 and TF-68.
- **The parked TF-58 branch** touches `TODO.md` only to remove its own row. No TF-68+ number is claimed elsewhere (checked 2026-10-03).

## Owner decisions — RULED (2026-10-03)

**Rulings:**
1. TF-68: **(A)**, probability models only.
2. TF-71: **include the marginal sensitivity model**; the row is Wicked.
3. TF-73: **route A**, plus a named opt-in.
4. TF-73: **(i′), refined** — one ball-centred (zero-mean) kernel for all three paths; σ fitted on the public StatsBomb open data with a pre-registered estimator; re-centred on each frame's ball; (ii) only if the OBSO predictive-validity gate fails.
5. TF-73: **(a)**, `scoring_prob_matrix` is an internal xT component, with the xG-boundary guard.
6. TF-73: **bundle under `silly_kicks/xthreat/`**; the G2 widening is a prerequisite.
7. **Keep the strict ordering rule**; TF-73 stays last in the Wicked band.

**The options as surfaced** (kept as the record):

1. **TF-68 population:**
   - (A) probability models only — Wicked;
   - (B) plus the ghost regressors with an in-house TreeSHAP — upper Wicked.
2. **TF-71 scope:** Size reads "Dunkin' (E-value only) / Wicked (+ marginal sensitivity model)", and its slot is provisional. Options:
   - include the marginal sensitivity model / Γ in TF-71 (resize to Wicked);
   - make it a separate row;
   - or do not pursue it.
3. **TF-73 delivery route:**
   - (A) default flip — breaking: version bump, Hyrum notice, new tokens, warning deprecated; the pre-existing `xt=` gap is moot;
   - (B) opt-in for EPV **and** transition — non-breaking only with transition (i) or a (i′) kernel gated to the opt-in (see decision 4); the pre-existing gap (`xt=` silences the warning while the transition stays synthetic) must be closed.
4. **TF-73 transition** (TF-73's Size cell and slot are conditional on it):
   - **(i) neutral all-ones static grid — small, value-changing.** Symmetric, no orientation split. On the pass path the transition drops out (OBSO = pitch control × EPV). `actual` and `peak` scale by a common per-pass constant (1 / the mid-pitch Gaussian's value at the target), so (i) changes `obso_actual` / `obso_peak` values; `pausa_temporal` and the frame that attains `peak` are unchanged. The shape change is confined to the teammate scan: `optimal` and `pausa_spatial` lose the mid-pitch weighting. The pass path stays ball-agnostic, as it is on `main`. Surface and space creation keep the ball kernel.
   - **(i′) all-ones plus a ball-anchored kernel in `compute_pass_obso` — moderate, value-changing.** Adds a ball term to the pass path; all three paths become ball-anchored, with no orientation split. A per-frame anchor makes the transition factor vary across the window's frames, so it no longer cancels in actual / peak: `pausa_temporal` changes, and so can the frame that attains `peak`. An event-frame anchor cancels in the ratio (spec-time choice).
   - **(ii) fitted destination-conditioned transition — large, toward Monstah.** It requires the deferred orientation-split reconciliation (value-changing) as a prerequisite, an API change, and a fit. Its ball-relative forms add a ball term to the pass path.
   - **Interaction with decision 3:** `compute_pass_obso` also serves explicit `transition_grid=` callers (`_obso.py:295`). Under route B, (i′) is non-breaking only if its kernel is gated to the opt-in. (ii)'s reconciliation changes values for callers passing an asymmetric `transition_grid=` under either route (latent on the symmetric default).
5. **TF-73 `scoring_prob_matrix`:**
   - (a) rule it an internal xT component — the bundle is an `ExpectedThreat`; it keeps `physical_grid` orientation handling; the route-B entry point is `xt="<name>"`;
   - (b) bundle a values-only grid through `epv_grid=` — P(goal \| shot) never ships; orientation must be handled for the raw grid; route B needs a new named slot on `epv_grid=`.
   - **Interaction with decision 3:** the route and this decision jointly fix the artifact, the route-B entry point and the route-A provenance token.
6. **TF-73 bundle placement / G2:**
   - place the bundle under `tracking/_<name>_weights/<variant>`, so the existing G2 glob covers it;
   - or depend on a G2-wide widening (combined-cycle work). Its cost: the per-bundle policy fields differ (`metadata.json` `shipped_variant` / `provider_list` vs `metrics.json` `providers` / `providers_trained` / `corpus_visibility`), and the flat `weights/` layout must be handled. (G2 checks corpus-policy keys and `run_tree_dirty`, not a commit field.)
7. **Ordering:**
   - keep the strict rule (TF-73 last in the Wicked band);
   - override it to place TF-73 first. The value argument: under route A it replaces the synthetic grids behind every default OBSO-family number on a shipped path except `pausa_temporal`, which is invariant to static grids (any per-frame-anchored ball term, whether (i′) or a ball-relative (ii), changes that one too); under route B it makes real surfaces available and closes the warning gap;
   - or add a value criterion to the On Deck ranking rule.

## Outside this batch (pre-existing; owner's call)

- `AGENTS.md:124` / `conventions-core.md:66` cited ADR-009 (the TF-24 Optuna harness) for the raw-primitives rule. Traced: the rule's only decision record is the TF-45 spec (`docs/superpowers/specs/2026-06-07-tf45-structural-pass-design.md`: rationale `:24-30`, scope `:90-95`, D4 `:149`), which keeps composites consumer-side as a mirror of ADR-009's frozen-exogenous-xT decision. The TF-45 commit (`da23a3f`) wrote it into `CLAUDE.md` with "(frozen-exogenous, ADR-009)"; the 2026-08-01 context-budget cut (`ebc5dab`) promoted it to a Key-conventions bullet and dropped the "frozen-exogenous" qualifier. No ADR recorded the rule. Revisions 1–7 called this a "mis-reference"; it is an analogy cite standing in for a missing ADR. The new rows cite the convention by name. **Addressed by ADR-110 (2026-10-04; see its Status):** it records the rule and its boundary, and the live docs now cite it.
- The `TODO.md` header said MCP Phase 2 was "in independent impl review"; it merged at `6cf8d82`. Fixed in revision 8 (owner-approved).
- The combined-cycle G2 test (`0cdcf05`) discovers only `tracking/` bundles; five non-tracking bundles are outside it. The author of this batch also reviewed that C1 and did not catch it. Ruling 6 makes the widening a TF-73 prerequisite, which closes this gap too.

## Review status

Reviewer B approved revision 6 (round 6, delta check, 3/3 passes); revision 7 adopted B's optional wording. Revision 8 applies the owner's rulings; that text has not been independently reviewed. All seven decisions are ruled.
