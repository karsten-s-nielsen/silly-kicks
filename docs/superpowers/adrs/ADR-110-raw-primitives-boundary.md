# ADR-110: The raw-primitives boundary — the library ships method-defined outputs; analyst choices stay consumer-side

| Field | Value |
|---|---|
| **Date** | 2026-10-04 |
| **Status** | Accepted (2026-10-04) |
| **Deciders** | Karsten S. Nielsen |

## Context

`AGENTS.md` lists as a key convention that the library ships RAW primitives, while composites, archetypes and rankings stay consumer-side. That convention, and the related rule that the consumer decides what a partial observation means, has been cited as ADR-009 in live docs, in code docstrings and in accepted ADRs. The citations known when this ADR was written are listed under Consequences, with a recipe for finding any others. ADR-009 does not record either rule. It records the TF-24 Optuna calibration harness. Its option C makes xT a frozen exogenous input, fitted on a disjoint corpus and fed to the calibration objectives (`ADR-009-calibration-harness.md:23`). That is an input to calibration, not a shipped default.

The rule's only decision record is the TF-45 spec (`docs/superpowers/specs/2026-06-07-tf45-structural-pass-design.md`: rationale `:24-30`, out-of-scope list `:90-95`, decision D4 `:149`). TF-45 ships the three structural primitives `structural_lbs` / `structural_sgm` / `structural_sdi`. It keeps the TIV z-norm composite, the K-means(4) archetypes and the ΔTIV / cumulative-TIV rankings consumer-side, because they "require a population reference … or a fitted clustering model". It describes this as mirroring the frozen-exogenous-xT decision of ADR-009. The TF-45 commit (`da23a3f`, 4.16.0) wrote the rule into `CLAUDE.md` as "**Library = RAW primitives only** — … stay consumer-side (frozen-exogenous, ADR-009)". The 2026-08-01 context-budget cut (`ebc5dab`) promoted it to a Key-conventions bullet and dropped the "frozen-exogenous" qualifier that marked the citation as an analogy. The 2026-09-25 restructure (#254, `14938af`) carried that bullet into `AGENTS.md`.

The rule's stated rationale, "the library stays stateless and pure; downstream consumers own the population statistics", does not describe practice when read literally:
- The library ships eleven bundled, corpus-fitted weights directories (`WEIGHTS_CLASSIFICATION`, `tests/test_bundled_weights_classification.py`), and the match-outcome default path requires its bundled ρ (ADR-097).
- It ships per-action composites (`rate_adjusted`, ADR-095; `rate_ximpact`, ADR-101; `packing_net`, TF-49), the TF-55 Glicko duel ratings, per-`(entity, match)` summaries, and per-entity pooling over a caller-chosen window.
- ADR-009's own constraint that the core stays "pure (pandas-in/out, zero I/O)" is about I/O; it does not forbid population-fitted parameters.

On 2026-10-03 the owner ruled that TF-73 will make a corpus-fitted xT the default EPV for the OBSO family, and that TF-70 (a team-adjusted player-skill posterior) lives in silly-kicks, with rankings consumer-side. New rows need a boundary that is written down and matches what the library does.

## Decision

The library ships outputs whose meaning is fixed by a method: raw primitives, frozen-parameter models, composites with derived or validated combinations, estimators with stated uncertainty, per-entity rollups, and validity verdicts on its own instruments. Analyst choices stay with the consumer: chosen weightings, descriptive structure fitted to the consumer's sample, interpretation policies for partial evidence, and rankings. A ranking that an identifiability gate licenses is a maintainer-reported artifact, never a library API.

**A. In the library (public API):**
1. **Raw per-action and per-frame primitives**, e.g. `structural_lbs` / `structural_sgm` / `structural_sdi` (TF-45), and per-action model outputs such as `goal_leverage` (ADR-101).
2. **Models with frozen, versioned parameters** fitted by the maintainers, shipped parameters-only and fail-closed (ADR-011), including corpus-fitted defaults. ADR-097's bundled ρ default is the precedent; TF-73's bundled xT applies it.
3. **Composites whose combination is derived or validated.** *Derived*: the combination follows from the method, with no weights chosen by anyone (`VAEP.rate_adjusted`, ADR-095; `VAEP.rate_ximpact`, `VAEP_adjusted × dP(win | goal)`, ADR-101; `pausa_composite`, temporal × spatial, the paper's own product of two ratios). *Validated*: the weights are frozen and have passed a validating battery (the condition ADR-104 option D sets for a threat + pressure blend). Weights merely chosen by an author or an implementation do not qualify, even when published; they are item 6.
4. **Estimators and validity verdicts.** Estimators fit a declared statistical model to caller-supplied data and return estimates with their uncertainty: the Glicko-2 duel ratings (rating + deviation, `duels/`, TF-55); TF-70's posterior. Validity verdicts are instrument- and construct-validity checks on the library's own outputs: `gkdv.behavioural_anchoring_verdict`, `gkdv.layer0_instrument_verdict`, `gkdv.layer1_responsiveness_verdict`, `xtgk.run_deep_zone_gate`. Neither is a ranking licence (item 10).
5. **Per-entity rollups**, at a fixed grain or over a caller-chosen window. Fixed grain: per `(entity, match)` — `gk_decision.summarize_gk_decision`, `positioning.summarize_positioning_gap`, `team_metrics.compute_team_kpis`, `territory.compute_territorial_dominance` and `duels.compute_duel_ratings` (each with `window=None`); per possession by default, or per match with `by="match"` — `restdefense.summarize_rest_defense`. Window: `gkdv.aggregate_by_keeper`, `duels.compute_duel_ratings(window=…)` (with `window_stat` choosing `as_of_end` or `change`), `territory.compute_territorial_dominance(window=…)`. A rollup is not a ranking.

**B. Consumer-side:**

6. **Chosen weightings**: weights picked by an author or an implementation rather than derived or validated. Examples: TIV's equal weights (the paper's own formula); a threat + pressure blend before it has frozen weights and a passing battery (ADR-104 option D). A weighting tool may ship for exploration (`positioning.WeightedSum`), but its output is not a glossaried default column.
   - **Named exception — `packing_net` (TF-49).** Its +1 / +0.5 / −1 direction multipliers come from the third-party `football-packing` library (TF-49 spec `:108-114`) and are frozen in `PackingParams`, but no validation of ours supports them. It shipped before this ADR, as a glossaried column of `add_packing`, and keeps shipping. The TF-74 battery decides its class: pass moves it to item 3; fail demotes it to opt-in, with a Hyrum notice and a version bump.
   - **Named exception — `xt_gk` and `xt_gk_rav` (xT-GK v1, ADR-024).** `xt_gk = T·(base + γ·PEV + RAV) + φ·DZV`, with T = η^k (`tracking/_xt_gk.py:284-292`), and `xt_gk_rav` = p·xT_dest − δ·(1 − p)·xT_counter (`:265-267`). `XtGkParams` sets γ = 0.25, δ = 0.55, φ = 1.0, η = 0.85, marked "interpretive / intent-set (NOT VAEP-calibrated)" and "PROVISIONAL (in-range)" (`:108-122`), plus per-style presets (`:140-144`). The author's source deck gives ranges for γ, δ and η (γ 0.1–0.4, δ 0.3–0.8, η 0.8–0.9) and none for φ; the implementation picked point values in range, and the author (Jeffrey Eyestone) accepted them as provisional (`:108-110`; ADR-024 `:43-45`; `NOTICE`). The weights are therefore chosen, not derived or validated. ADR-036 (M5, `:71`) froze v1 alongside v2; both columns are glossaried and emitted by `add_xt_gk` / `compute_xt_gk`, and keep shipping. Their class is decided with the author, in the TODO item "xT-GK v2 interpretation-fork decision + v1 preset status": validate the presets — a sensitivity check across the deck ranges for γ, δ and η and a range agreed with the author for φ, plus the xT-GK construct-validity harness (`docs/research/xtgk_v2_construct_validity/`, built for v2) — to move them to item 3, or keep the named exception. The exception also ends if v1 is removed, which ADR-036 (M5) schedules for no earlier than one release after the lakehouse migrates its `xt_gk*` columns.
   - **These are the named exceptions known at writing; the list is not closed.** They were found by grepping the glossary for composite / weighted / blend wording, the library for "intent-set", "PROVISIONAL" and "never calibrated", and parameter classes for weight-like fields, not by a column-by-column audit. Weights taken from model outputs (the threat- and completion-weighted columns) are derived. Thresholds and shape / scale constants of a single primitive's transform are not combination weights, so they do not make a column item 6: e.g. xT-GK's `dzv_alpha` = 2.1 and `dzv_beta` = 0.8 ("CANONICAL (Eyestone 2026-06-27)"), `dzv_d_max` ("provisional"), `defensive_third_boundary` and `pressure_scale` ("intent-set") (`tracking/_xt_gk.py:111-113`, `:123-128`), and the defensive-credit and press-commitment constants (`tracking/defensive_credit/_params.py:92-104`; `tracking/_press_commitment.py:51-55`). Their calibration status is recorded where they are defined; this ADR does not classify it.
7. **Descriptive structure fitted to the consumer's sample without a validated meaning**, e.g. K-means archetypes, and z-normalisation against the consumer's population.
8. **Interpretation policies for partial evidence.** The library ships the raw coverage: the visibility polygon as raw data (ADR-054); opt-in, additive visibility companions (ADR-062); `point_observed` / `region_observed_fraction` (ADR-055). What a partial observation means for a count or a decision stays with the consumer.
9. **Rankings and leaderboards**, i.e. ordering entities by a metric, as a library API.

**C. Reported artifacts (maintainer-produced under `docs/research/`, never a library API):**

10. **A licensed ranking.** A ranking of entities is licensed only by an identifiability gate that separates the entity from its team: the crossed entity + team ICC (ADR-099, ADR-104), or an equivalent mobility / connectedness check. A licensed ranking, and the identifiability verdict that licenses it, ship as a reported artifact through ADR-009's recommend-then-apply process; ADR-099's `docs/research/territory_ranking_census/ranking.parquet` is the precedent.
    - The per-entity metric **may** ship ungated, decided per case: ADR-099 shipped its per-`(defender, match)` metric ungated, while ADR-104 (`:51`) deferred per-defender attribution behind the gate. TF-55's ratings ship ungated as estimator outputs; no duel-rating ranking has been licensed, and a ranking a consumer builds from an ungated metric carries no library endorsement.
    - Only the identifiability verdict that licenses a ranking is a reported artifact. Instrument- and construct-validity verdicts on the library's own outputs are not ranking licences; they are item 4.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Repoint the citation to the TF-45 spec only | One-line change | Leaves the boundary unrecorded, and the literal "stateless" rationale contradicts eleven shipped bundles, the shipped composites, the duel ratings, per-entity rollups and TF-73 | Fixes the pointer, not the gap |
| B. A literal stateless rule: nothing corpus-fitted in the library | Matches TF-45's wording | Would forbid ADR-011 artifacts, ADR-097's bundled ρ, TF-55, TF-70 and TF-73 | Contradicts settled practice and owner rulings |
| C. Everything in-library if documented, including chosen weightings and rankings | Maximum convenience for consumers | Ships analyst choices as if they were measurements; rankings inherit the team confound that ADR-099 / ADR-104 gate | Turns analyst judgment into library defaults |
| D. Method-defined meaning (chosen) | Matches each shipped precedent listed under Notes, apart from the named exceptions (`packing_net`; `xt_gk` / `xt_gk_rav`); gives new rows one test to apply | "Derived or validated" still needs review at the edge | — |

## Consequences

### Positive

- The convention has a decision record, and the live docs cite it.
- TF-73's bundled default (item 2), TF-70's estimator (item 4; rankings consumer-side and contrast posteriors only, per the owner's ruling) and TF-68's attributions (item 1: per-prediction outputs of shipped models) are explicitly in bounds.
- Reviewers have one question to ask of a new output: is its meaning fixed by a method, or is it an analyst choice?

### Negative

- The boundary needs judgment at the edge, e.g. whether a combination is derived or chosen. A review has to answer that per case.
- Shipped columns stay named exceptions until their deciding work runs: `packing_net` (TF-74) and `xt_gk` / `xt_gk_rav` (the xT-GK decision with its author). Other chosen-weight composites may surface; each is classified when found.
- **ADR-009 is cited in two senses, and only one moves here.**
  - **Still ADR-009: its own decisions and their applications.** Frozen `*Params` with an empty `for_provider` map; reported-not-gated recommendations that never change a library constant; the recommend-then-apply process ("a future ADR-009 gated on a crossed ICC", ADR-104 `:51`, `docs/context/gk-metrics.md`, `tracking-metrics.md`, `scripts/validate_territorial_defense.py`; "an ADR-009 apply", `docs/context/event-metrics.md`, the second half of ADR-099 `:80`; "tuning … gated owner concerns", the second half of ADR-092 `:160-161`); and the calibration fallback (`docs/context/event-metrics.md`, ADR-095 `:23`).
  - **This rule, previously cited as ADR-009, in accepted ADRs.** These are records and are not rewritten; read these citations as ADR-110: ADR-054 `:115`; ADR-055 `:256`, `:286`; ADR-062 `:33`; ADR-092 `:160-161` (first half); ADR-099 `:80` (first half); ADR-101 `:28`, `:39`; ADR-104 `:30` and the "raw primitives" half of `:58`.
  - **This rule, in live docs and docstrings, now repointed.** To ADR-110: `AGENTS.md`; `docs/context/conventions-core.md` (`:56`, `:66`); `docs/context/tracking-metrics.md`; `silly_kicks/tracking/_visibility.py` (`:11`, `:252`); `silly_kicks/tracking/features.py` (the `visible_area` note of `add_action_context`); `scripts/_crossed_icc.py`; and `silly_kicks/tracking/_structural_pass.py`, whose TF-45 docstring keeps the ADR-009 analogy beside ADR-110. To ADR-077, the decision that made these aggregators' `visible_area` companions opt-in and additive (extending ADR-062's `add_action_context` companions): the eight "opt-in and additive" FOV-companion docstrings (`silly_kicks/tracking/features.py`, seven; `tests/tracking/test_fov_companions.py`, one).
  - **To find any others:** `git grep -n "ADR-009"`. A citation belongs to this rule when it says the library leaves composites, rankings or interpretation policy to the consumer, or ships something "raw"; otherwise it is ADR-009's own sense. `docs/superpowers/specs/`, `docs/superpowers/plans/`, `CHANGELOG.md` and the frozen fixture `tests/fixtures/claude_md_at_77286f4.md` are history and stay as written.

### Neutral

- No behaviour changes. Citations change in `AGENTS.md`, `docs/context/conventions-core.md`, `docs/context/tracking-metrics.md`, the docstrings listed above, and the `raw-primitives-consumer-side` entry of `tests/fixtures/agents_md_invariant_inventory.json`. `conventions-core.md` and the inventory keep ADR-009 named as the analogy TF-45 invoked. `docs/research/research-batch-2026-10/VETTING.md` records the trace; `TODO.md` gains TF-74 (the `packing_net` battery), and its item "xT-GK v2 interpretation-fork decision" is amended to "… + v1 preset status" (the `xt_gk` / `xt_gk_rav` presets). The `summarize_positioning_gap` docstring example no longer sorts teams into a ranking (item 9). Owner-approved fold-ins outside this rule: the `scripts/validate_match_outcome_calibration.py` docstring is corrected (since ADR-097 the default path uses the bundled ρ), and `TODO.md` gains TF-75 (re-run the TF-53 calibration study so its provenance reflects the ADR-097 defaults).

## Related

- **Specs:** `docs/superpowers/specs/2026-06-07-tf45-structural-pass-design.md` (`:24-30`, `:90-95`, D4 `:149`); `docs/superpowers/specs/2026-07-16-tf49-packing-design.md` (`:108-114`, `packing_net`)
- **ADRs:** ADR-009 (the analogy TF-45 invoked: frozen exogenous xT); ADR-011 (parameters-only, fail-closed artifacts); ADR-097 (bundled ρ default); ADR-054, ADR-055, ADR-062, ADR-092, ADR-095, ADR-099, ADR-101, ADR-104 (applications)
- **Commits:** `da23a3f` (TF-45, 4.16.0: wrote the rule into `CLAUDE.md` with "(frozen-exogenous, ADR-009)"); `ebc5dab` (2026-08-01 context-budget cut: promoted it to a Key-conventions bullet and dropped "frozen-exogenous"); `14938af` (#254, `AGENTS.md` restructure)
- **TODO:** TF-74 (`packing_net` validation battery); "xT-GK v2 interpretation-fork decision + v1 preset status" (the `xt_gk` / `xt_gk_rav` presets); TF-75 (TF-53 calibration re-run, an owner-approved fold-in)
- **ADRs (xT-GK):** ADR-024 (xT-GK v1); ADR-036 (`:71`, v1 frozen alongside v2)
- **Research:** `docs/research/research-batch-2026-10/VETTING.md` ("Outside this batch")

## Notes

The applications in force when this ADR was written:

| Case | In the library | Consumer-side or reported |
|---|---|---|
| TF-45 structural pass | `structural_lbs` / `structural_sgm` / `structural_sdi` (item 1) | TIV composite: chosen equal weights and a population z-norm (items 6, 7); K-means archetypes (item 7); ΔTIV rankings (item 9) |
| ADR-095 xSuccess | `VAEP.rate_adjusted` (item 3, derived) | — |
| ADR-101 xImpact | per-action `goal_leverage` (items 1, 2); `VAEP.rate_ximpact` (item 3, derived) | season / player ranking aggregate (item 9) |
| ADR-104 positioning | threat-only `positioning_gap` (item 1); `summarize_positioning_gap` (item 5) | threat + pressure blend until frozen weights and a battery (item 6); `WeightedSum` for exploration; per-defender attribution / ranking deferred behind the gate (`:51`) |
| TF-54 territory (ADR-086) | `compute_territorial_dominance`, per-`(player, match)` or windowed (item 5) | rankings (item 9) |
| ADR-099 territorial "threat prevented" | `method="counterfactual"` per-`(defender, match)` metric (item 5), with the bundled `PassCompletionModel` (item 2) | the licensed defender ranking and its census verdict, a reported artifact (item 10) |
| TF-55 duels | Glicko-2 ratings with deviation (item 4); windowed rows (item 5) | leaderboards (item 9); no ranking licensed |
| TF-49 packing | `packing_made` (item 1) | `packing_net`: named exception under item 6, shipping until TF-74 decides |
| xT-GK v1 (ADR-024; frozen by ADR-036) | `xt_gk_base`, `xt_gk_pev`, `xt_gk_dzv`, `xt_gk_pressure`, and the coordinate / provenance columns (`xt_gk_origin_*`, `xt_gk_dest_*`, `xt_gk_*_source`, `xt_gk_origin_confidence`, `xt_gk_completion_variant`, `xt_gk_native_goalkick_out_of_region`) (item 1) | `xt_gk` and `xt_gk_rav`: named exception under item 6 (provisional values chosen in the author's ranges), shipping until the xT-GK decision with its author or v1's removal (ADR-036 M5) |
| GK decision, rest defence, team KPIs | `summarize_gk_decision`, `summarize_rest_defense`, `compute_team_kpis` (item 5) | rankings (item 9) |
| GKDV | `aggregate_by_keeper` (item 5); instrument verdicts (item 4) | keeper rankings (item 9) |
| PAUSA | `pausa_composite` (item 3, derived) | — |
| Visibility (ADR-054 / 055 / 062) | raw coverage, opt-in companions (item 1) | what a partial observation means (item 8) |
| TF-73 (planned) | bundled corpus-fitted xT as the default EPV (item 2) | — |
| TF-70 (planned) | team-adjusted player-skill posterior with uncertainty (item 4); contrast posteriors | rankings (item 9; owner ruling 2026-10-03: "never a ranking") |
