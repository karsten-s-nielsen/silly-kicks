# xT-GK multi-provider GK-completion — provider-aware variant design (design spec)

## Executive summary (for review)

xT-GK's RAV term needs `P(success | geometry)` from a fitted `GkCompletionModel`. The 4.21.0 release ships a **GS-trained `default`** (native-completion label, AUC 0.838). SkillCorner is the more common provider in the wild — "most people do not have GS" — so a GK-completion model that works on SkillCorner is what makes xT-GK broadly usable. This spec settles **how** to serve SkillCorner (and the other providers) **based on measurement, not assumption**, and — after review round 1 — carries the construct difference **through to the consumer**, not just training.

**The decisive measurement (real data, both probes 2026-06-09, post-4.20.1-bugfix):**

| served model | corpus | native OOF AUC | notes |
|---|---|---|---|
| **GS `default` → SkillCorner** | 8 matches, N=460 | **0.500** (chance) | GS does **not** transfer; `−length` AUC 0.42 (length sign *inverts* on SC) |
| **SkillCorner-trained (own 5-fold CV)** | 10 matches, N=547 | **0.696** overall | learnable on SC — but only for the GK-pass sub-domain |
| ↳ SkillCorner-trained, **GK-passes** | N=459 | **0.741** | the usable signal |
| ↳ SkillCorner-trained, **goal-kicks** | N=86 | **0.478** (chance) | geometry not predictive of SC's possession-proxy label here |
| GS `default` → IDSSE | (prior, §1) | **0.824** | transfers (kloppy sportec gateway infers completion) |

**Five facts the design respects:**

1. **The GS→SkillCorner non-transfer (0.50) was a LABEL bug, not a geometry/transfer truth.** 4.20.1 fixed the SkillCorner converter (time-base + goal-kick label); the GS model still served at chance (0.50) on the *fixed* data — because the SkillCorner converter's `result_id` was a `same_team_next` **possession proxy**, a *different target* from GS native completion. **The spike (§4) then found SkillCorner DOES carry a native `pass_outcome` outcome** the converter simply didn't surface. So the fix is to route `result_id` onto native completion (D-S8); once both providers share the native-completion construct, the 0.50 is **void** and GS-transfer is re-measured (D-S1). (Earlier framing — "SkillCorner has no native outcome" — was **false**; corrected here.)

2. **SkillCorner is learnable — for GK-passes, with its own model** (0.741). So the answer is a **per-provider variant**, not "SkillCorner unsupported."

3. **SkillCorner goal-kicks are not learnable from geometry against the proxy label (0.478, chance).** Served as a tagged per-type base rate — an honest limitation, never dressed as a prediction.

4. **(Review H1) The proxy `p` is a *different quantity* than the `p` RAV consumes — and that must be carried through to the consumer, not just training.** RAV is `p·xT★(z′) − δ(1−p)·xT★(counter)`, where `p` must mean *P(the distribution reaches the intended destination z′)* — that is why it weights `xT★(z′)`. SkillCorner's proxy is *P(retain possession)*. So SC `xt_gk` is **"retention-weighted threat"** while GS `xt_gk` is **"completion-weighted threat"** — the **same column name, two different constructs**. The variant split fixes *training*; the spec now also fixes *consumption* (a hard "do not pool `xt_gk` across variants" contract + provider-local qualification, §2.5).

5. **(Review C1) For a probability consumed *multiplicatively* in RAV, absolute calibration matters more than discrimination.** AUC gates ordering; RAV needs scale. Calibration (ECE / reliability-slope within tolerance) is now a **hard gate** for every variant whose output is consumed as a probability (§3.2) — alongside, and ahead of, the AUC floor.

**Recommendation (synthesized) — feasibility now CONFIRMED, so this is no longer a fork.** Ship **per-provider completion variants** selected automatically from `frames["source_provider"]` (with a `completion=` override and a documented `gs`-fallback-with-warning), **calibration-hard-gated on a common scale with GS**, **comparability-validated before pooling**, and emitting the `xt_gk_completion_variant` provenance column **even though consumption pools** (the only post-hoc detection/repair seam). All variants train on **`result_id`**, made uniform by fixing SkillCorner `result_id` to the **single native-completion construct** — **`pass_outcome` primary** (the SPADL "reached a teammate" contract), `received==True` as a *success-only* augmentation, and a **flagged** construct-correct inference for the residual gap (never `received==False→fail`, never a silent proxy — review N1). So SkillCorner `p` is the **construct-correct** quantity RAV wants (P(reaches target), comparable to GS), with no side-channel column and no blended label.

**The spike result (measured 2026-06-09, 4 SkillCorner matches, 19,835 events) — decisive and positive:**
- **`pass_outcome` (`successful`/`unsuccessful`) is native** — on goal-kicks **59.6% coverage, non-degenerate (~75% success)**; on GK-actor passes ~35% (~81% success). `received` (targeted player received) covers GK-actor passes 64% (absent on goal-kicks). `player_targeted_id` is a **native receiver identity** (95% on goal-kicks) — the S1 attribution-noise concern is **moot** (no nearest-player attribution needed).
- **The current `same_team_next` proxy is construct-*wrong*, not merely different:** on GK-actor passes (n=388) native `received_rate`=0.43 vs proxy `same_team_next_rate`=0.84, **agreement 0.459 ≈ statistical independence**. The proxy carries almost no information about whether the GK's distribution reached its target — which is exactly why SkillCorner-CV capped at 0.74 and goal-kicks sat at chance (0.478): **the label was wrong, not the geometry.** Reception isn't a nicety; it fixes a wrong label, and pooling SkillCorner-on-proxy `xt_gk` with GS would have been *actively misleading*.

This collapses round-2's S1/S2 (the expensive attribution branch and its timebox are unneeded) and the lakehouse's Consequence-2 release risk (the spike's cheap half succeeded). What remains from the lakehouse decision: the **cross-provider comparability gate** (Consequence 3) and **common-scale calibration** (Consequence 4), both folded in below.

**Label sourcing — GOLD STANDARD (D-S8, settled by the maintainer 2026-06-09): fix SkillCorner `result_id` to native completion, accept the VAEP retrain.** The `same_team_next` proxy is a SkillCorner **converter correctness bug** (BUG 5) that mis-labels completion for *every* `result_id` consumer (VAEP, all features, the lakehouse SPADL), not just xT-GK. So the root fix is to route SkillCorner pass/set-piece `result_id` through the native outcome — **`pass_outcome` where present → `received` → `same_team_next` only as the residual coverage fallback** — making completion construct-correct *everywhere*, one source of truth. The xT-GK completion model then reads `result_id` **uniformly for all providers** (no SkillCorner side-channel column, no per-variant label divergence). This is a SkillCorner-wide **VAEP-retrain trigger** (scores/concedes label distribution shifts) + **golden-test updates** — accepted: the maintainer prefers one more retrain for the correct long-term architecture, and the lakehouse waits until xT-GK is ready before consuming. (The earlier dedicated-`native_completion`-column idea was a scope-minimizing band-aid to avoid the retrain; rejected now that scope is not a constraint — it would leave the converter bug in place.) **Chesterton's Fence:** the build verifies the converter spec + `git blame` for *why* `same_team_next` was chosen (it may cover non-pass/defensive rows `pass_outcome` does not) before replacing it; the fallback chain preserves 100% coverage.

**Architectural consequence — the per-provider *weights* question is now an open re-measurement, not an assumption.** The GS→SkillCorner non-transfer (AUC 0.50) that motivated provider-specific weights was measured on the *proxy* label and is **void**. On the corrected native label (same construct as GS), GS may transfer — so the build **re-measures** GS-transfer on the corrected label: if it clears the gate, the `skillcorner` variant can be the GS model (or a thin re-fit) and no provider-specific weights are needed; if not, SkillCorner-specific weights stay. The **variant-selection architecture stays regardless** (the pooling-safe, extensible seam carrying the comparability gate + provenance); only whether SkillCorner needs distinct *weights* is re-opened (D-S1).

**Scope — SETTLED: single 4.21.0 with everything**, now including the SkillCorner `result_id` completion fix as the foundational change. xT-GK + GS goal-kick coverage + corrected SkillCorner completion + the (re-measured) SkillCorner variant ship together; one VAEP retrain covers the lot.

---

**Date:** 2026-06-09 · **Author:** Karsten (with Claude) · **Status:** **APPROVED FOR IMPLEMENTATION** — review rounds 1+2+4 resolved (§10/§11/§13); reception spike done + lakehouse pooling decision folded in (§12); D-S8 settled to the gold-standard `result_id` fix (maintainer 2026-06-09); rev-4 N1 (single-construct label, not a blend) + N2 (contradictions) + N3 (sequencing) + N4 (gate contingency) folded in. All decisions closed. Next: write the implementation plan.
**Context:** extends `GkCompletionModel` (from the goal-kick-coverage spec `2026-06-08-xt-gk-goalkick-coverage-design.md`, which settled the GS `default` + `_resolve_gk_geometry` + per-type base-rate self-sufficiency) from a single GS artifact to a **provider-aware variant family**. Builds on the 4.20.1 SkillCorner converter fixes (`reference_skillcorner_converter_data_bugs`) — the measurements here are on *fixed* SkillCorner data. Folds into silly-kicks 4.21.0, ADR-024.
**Attribution:** within Jeffrey Eyestone's delegated "derive P(success) from the library's pass-completion machinery" (xT-GK is public-with-attribution, Rule #1 lifted). No new external methodology — same logistic family, re-fit per provider. FYI to Jeffrey; not a blocker.

---

## 1. Evidence (real data, both probes 2026-06-09, on 4.20.1-fixed converters)

**Probe A — GS `default` served on SkillCorner (8 matches, N=460, FIXED data).** Confirms the bug fixes landed *and* that transfer still fails:

| metric | value | reading |
|---|---|---|
| base rate | 0.807 | non-degenerate (BUG-2 fixed: was 1.000) |
| density finite, P1 / P2 | 100% / 100% | BUG-1 fixed (was 100% / 12% — time-base) |
| **native model AUC** | **0.500** | GS model serves at **chance** on SC |
| `−length` AUC | 0.418 | length anti-predictive on SC's label (GS: length is the dominant −coef) |
| goal-kick AUC / gk_pass AUC | 0.562 / 0.470 | neither sub-domain works with the GS model |

**Probe B — SkillCorner-trained, own 5-fold match-grouped CV (10 matches, N=547):**

| sub-domain | N | SkillCorner-CV AUC | GS-transfer AUC |
|---|---|---|---|
| overall (native) | 547 | **0.696** | 0.521 |
| **GK-passes** | 459 | **0.741** | 0.487 |
| goal-kicks | 86 | 0.478 (chance) | 0.581 |

SkillCorner full-fit coefficients (standardized): `length −0.866`, `dy_abs +0.857`, `dest_y_off +0.423`, `forwardness +0.388`, `is_throw_in +0.204` *(inert — see review m3: near-empty GK-throw-in positive class; not signal)*, `dest_x +0.135`, `dest_defender_density −0.103`, `is_goalkick −0.088`.

**Prior (goal-kick-coverage spec §3.3, R2):** GS `default` transfers to **IDSSE at AUC 0.824** — `_loader_pining._build_idsse` uses the **kloppy sportec gateway**, which *infers* a native-style completion label. IDSSE shares GS's label construct.

**Construct root cause (per-provider target variable):**
- **GS / StatsBomb / Wyscout / IDSSE-via-kloppy / sportec-direct:** native completion — *did this pass reach a teammate* (real outcome event / DFL `play_evaluation`).
- **SkillCorner:** the converter's `result_id` *currently* uses a `same_team_next` **possession-retention proxy** — but a **native completion outcome exists in the raw data** (`pass_outcome`/`received`, §4) and is simply not surfaced. The gold-standard fix routes `result_id` onto it (D-S8), making SkillCorner the **same native-completion construct as GS**.
- **Metrica:** untested for xT-GK (converter defaults result to success, loss via separate BALL-LOST events); treat unsupported until measured.

## 2. Design — provider-aware completion variants

### 2.1 Variant family (not a blend)

All variants train on **`result_id`** — uniform across providers once SkillCorner `result_id` is corrected to native completion (D-S8/§4). The registry maps provider → **weights**; whether SkillCorner needs *distinct* weights from `gs` is re-measured on the corrected label (D-S1), but the same logistic envelope (coefficients + `feature_names` + standardization stats + SHA256, no pickle) and the same label *source* (`result_id`) apply to all:

| variant key | label source | trained on | distinct weights? |
|---|---|---|---|
| `gs` (= current `default`) | `result_id` (native completion) | GS WC2022 (30 matches) | yes (bundled 4.21.0) |
| `skillcorner` | `result_id` (native completion, **corrected** via D-S8) | SkillCorner pining corpus | **re-measure GS-transfer on the corrected label (D-S1)** — GS model if it transfers, else SkillCorner-specific weights |
| (`metrica`) | — | — | no (unsupported until measured) |

No multi-provider blended model — the measured non-transfer (0.50) proves a blend would average two incompatible targets.

### 2.2 Serve-time variant selection

Selection precedence:

1. **Explicit `completion=` override** (a `GkCompletionModel`) — always wins. **This is also the documented escape hatch for cross-provider setups** (review C2): the completion *label* lives in `actions.result_id` (the **event** provider), while `frames["source_provider"]` is the **tracking** provider. Auto-selection assumes event-provider == tracking-provider (the common single-stack case, e.g. SkillCorner events + SkillCorner tracking). For a mismatched stack (e.g. StatsBomb events + SkillCorner tracking → native completion), pass `completion=GkCompletionModel.from_variant("gs")`.
2. **Auto from `frames["source_provider"]`** via the **pure** `variant_key_for_provider` (review C4). **Why key on the *tracking* provider when the label construct is an *event*-provider property (review m-a):** SPADL `actions` carry **no** provider column, so `frames["source_provider"]` is the only available provider signal. It is a **proxy** for the event/label provider — exact in the single-stack case, and `completion=` (precedence 1) covers the mismatched-stack case. The mapping: `skillcorner → "skillcorner"`; `{gradientsports, sportec, snapshot} → "gs"`; unknown/`None`/`metrica → "gs"` + warning (`stacklevel=2`, sharper "untested for xT-GK completion" wording for `metrica`). `sportec → gs` is verified construct-correct: `spadl/sportec.py:855–858` derives native completion from DFL `play_evaluation` (`unsuccessful → fail`), non-degenerate (~71% goalkick completion). `snapshot → gs` because the completion label rides the actions (event) provider, which for snapshot frames is typically a native-completion event source (e.g. StatsBomb freeze-frames).
3. The resolved key is recorded in provenance (§2.4).

**Mixed-provider handling (review C3, refined).** Compute the set of **distinct non-null real** `source_provider` values (excluding `snapshot`, which is a frames-only synthetic tag and never a standalone real provider). If **two or more distinct real providers** appear → **raise** (one xT-GK call is one match = one provider; multiple = a linkage/ingestion bug). A lone `snapshot` (or `snapshot` + exactly one real provider) does **not** raise — verified: `_snapshot.py` builds a complete all-`snapshot` frames table per match (never appended to a provider's frames), so no legitimate pipeline mixes them today; the exclusion is cheap insurance against a future augmentation path.

### 2.3 SkillCorner goal-kick handling (re-measured on the native label)

The earlier chance result (0.478) was on the *proxy* label; goal-kicks now have a **native, non-degenerate** `pass_outcome` label (§4), so the build **re-measures** whether geometry predicts native goal-kick completion. Two outcomes, both tagged via `xt_gk_completion_source`:
- **If native goal-kick AUC clears the floor →** model-scored (`xt_gk_completion_source = model`).
- **Else →** the **per-type calibrated base rate** (self-sufficiency path, goal-kick spec §3.3 R5), tagged `base_rate` — an honest, filterable fallback, never a fabricated prediction.
- `xt_gk_origin_confidence` (already emitted) further down-weights base-rate rows.

**m2 (label tier ≠ inference path):** `result_source` (`native`/`inferred`/`stopgap`) is a *training-label-quality* tier, NOT an inference gate. At inference the model scores **any row with geometry**, whatever its label tier would have been (the tier only weights/filters training). Base-rate-served-at-inference is reserved for **geometry-missing** rows only (the existing self-sufficiency path) — never for label-tier reasons.

### 2.4 Provenance additions

On top of `xt_gk_origin_source` / `xt_gk_dest_source` / `xt_gk_origin_confidence`:
- `xt_gk_completion_variant ∈ {gs, skillcorner}` — which variant scored the row.
- `xt_gk_completion_source ∈ {model, base_rate}` — model prediction vs per-type base-rate fallback.

`XtGkReport` gains per-variant + per-completion-source counts.

### 2.5 Consumer-side comparability contract (review H1 — NEW)

The variant split fixes training; this contract fixes **consumption**. Two constructs share the `xt_gk` column name, so:

**Both variants now share ONE construct (native completion / reach-z′) once `result_id` is fixed (D-S8).** So H1's "two different quantities under one column name" is resolved at the *label* — pooling is construct-valid in principle. The residual cross-provider risk is **scale**, not construct: provider-systematic geometry/coverage offsets (SkillCorner goal-kicks ~17 m median vs GS, tracking-extrapolated vs event-derived coords, different `receiver_zone_density` distributions) can shift the `xt_gk` distribution even when both labels mean reach-z′. The contract therefore shifts from "never pool different constructs" to:

- **Pool only after the comparability gate passes (D-S9).** Documented in the column docs, `XtGkReport`, and ADR-024. Until the gate validates cross-provider scale (or a validated re-scaling is applied), `mean(xt_gk)` over mixed `xt_gk_completion_variant` is not yet sanctioned.
- **Runtime signal (review m-c):** the contract is *advisory* in-library (real enforcement is the consumer's). `XtGkReport.spans_multiple_variants` (true when scored inputs span >1 variant) makes a mixed-variant aggregation machine-observable rather than doc-only.
- **Keep `xt_gk_completion_variant` in `fct_action_context` even though consumption pools (lakehouse rec 3)** — the only post-hoc detect/repair seam, and it lets the lakehouse DQ checks assert "one (validated) construct per pooled aggregation" if comparability ever regresses.
- **Lakehouse decision — SETTLED (2026-06-09): `xt_gk` IS pooled across providers** in `fct_action_context`. So an interim retention-proxy was never viable (it would pool two constructs with no enforcement seam — H1 in production); the D-S8 native-`result_id` fix is what makes pooling legitimate.

**D-S9 comparability gate — concrete contingency (review N4, since pooling is locked):** compare SkillCorner vs GS `xt_gk` distributions on matched distance / pitch-zone bands (owner/public-run), with a stated **minimum overlap-band n** (the SC-short / GS-long goal-kick distance overlap is thin → require ≥ N rows per compared band or the band is reported as under-powered, not silently passed). Three outcomes, **no silent pooling on an offset scale**:
  1. **Within tolerance →** pool directly.
  2. **Offset with positive evidence it is a measurement/scale ARTIFACT** (systematic *and* uniform across bands, consistent with a known tracking/calibration difference — **not** merely "the distributions differ"; review F4) → apply a **documented affine re-scaling**. **Located on `xt_gk` itself, NOT on `p` (review G2):** `xt_gk` is nonlinear in `p` and depends on the threat-grid terms, so an affine on `p` (in `predict_proba`) neither maps to a known `xt_gk` affine nor is safe (it would undo the certified common-scale `p`-calibration). The re-scale is therefore a **per-variant post-composite affine on `xt_gk`** in `_xt_gk.py`, **clamped to a sane range** (m5), baked into the silly-kicks output so the lakehouse pools raw. A genuine football difference (SC ~17 m goal-kicks) must **not** be re-scaled away.
  3. **Ambiguous / large / unstable / genuine-football offset →** **escalate to the maintainer** (**the default** — after common-scale `p`-calibration a residual `xt_gk` offset is, by elimination, the threat-term difference = genuine football, so `correctable` is rare-to-nonexistent and `within_tolerance`-or-`escalate` is the expected outcome; review G2). Conforming SC to GS is the *exception requiring evidence*, not the default.
- **Common-scale calibration (lakehouse Consequence 4):** the C1 gate is checked against the GS variant's reliability target, not per-variant in isolation (§3.2).

## 3. SkillCorner variant — training & gate

### 3.1 Corpus & label
- **Corpus:** SkillCorner slice of `_loader_pining` (public); the bundled variant trains on the full available SkillCorner pining set.
- **Label:** **`result_id`** — uniform with GS, now construct-correct for SkillCorner because the converter `result_id` is fixed to native completion (`pass_outcome` → `received==True` success-only → residual `same_team_next`, D-S8/§4), tiered via `result_source ∈ {native, inferred, stopgap}`. **Training uses the `native` tier ONLY (review F1 + G1):** `prepare_gk_completion_training_data` keeps only `result_source == "native"` (`pass_outcome`) rows. `inferred` (`received==True`, etc.) is **positive-only** (clean successes, no clean fails) → including it would push the training positive rate above the true completion rate → bias the logistic intercept high → **mis-calibrate `p`**, the primary hard gate consumed multiplicatively in RAV; only `pass_outcome` supplies **both** classes from one rule. `result_id` keeps `inferred`/`stopgap` values for VAEP coverage (provider-agnostic — GS has no `result_source` → no-op). **Model card** documents the native-field provenance + per-tier coverage.
- Features: identical `extract_gk_completion_features` set; train==serve parity preserved (same code path, same `_resolve_gk_geometry`).

### 3.2 Gate — calibration is a hard gate (review C1), common-scale across variants (both now native completion)
For **any** variant whose output is consumed as a probability in RAV:
- **Calibration — HARD GATE (primary):** ECE ≤ tolerance **and** reliability-slope within `[1−ε, 1+ε]` on held-out (match-grouped). RAV multiplies `p` by threat magnitudes, so a well-ordered-but-mis-scaled model (high AUC, bad calibration) produces systematically wrong RAV magnitudes. This gate ranks **ahead of** the AUC floor. **Common-scale check (lakehouse Consequence 4):** because `xt_gk` is pooled across providers, the SkillCorner variant's calibration is checked against the **same reliability target as the GS variant**, not just per-variant in isolation — a SkillCorner model well-calibrated to its own outcome but on a different scale than GS still distorts a pooled GK ranking. **(review m-b, now favourably resolved):** because the SkillCorner variant trains on **native completion** (reach-z′), calibration *is* to the RAV-relevant construct — not to a proxy. The residual cross-provider concern is **scale** (handled by the common-scale check + the §2.5 comparability gate), not a construct mismatch. (Had we shipped the proxy, "calibrated" would have meant well-scaled to retention, not reach-z′; the native label removes that trap.)
- **AUC floor (secondary):** `gs` ≥ 0.84 (native-origin held-out); `skillcorner` ≥ **0.70 on GK-passes**, **re-measured on the native label** (the 0.741 measured earlier was on the *proxy* label and does not bind — the native label is expected to be at least as learnable, since the proxy was near-uninformative). 0.70 is a real non-trivial floor, not a rubber stamp. **Goal-kicks: re-measured on the native `pass_outcome` label** (non-degenerate, ~75% success) — NO LONGER auto-excluded; the model-vs-base-rate switch uses a **lower-confidence-bound** AUC threshold **and reports n** (review m3 — the goal-kick sample is a few hundred rows × ~0.6 native-label coverage, so a point-estimate switch is noisy). The proxy's 0.478 does not bind the native label.
- **Owner/public-run, not a PR CI gate (review m2):** SC pining is owner-tier (like the GS gate). The gate is the *training-time green criterion*; CI exercises synthetic fixtures (§7), not the owner corpus.

### 3.3 Serialization & bundling
Same JSON envelope as `gs`. Bundled at `silly_kicks/tracking/_gk_completion_weights/skillcorner/`. `scripts/train_gk_completion.py` gains `--variant skillcorner` (provider-scoped corpus + native-completion label + the calibration/common-scale + GK-pass gate). `from_variant("skillcorner")` loads it; `from_variant("default")` stays `gs`.

## 4. SkillCorner native completion label — CONFIRMED (spike done, 2026-06-09)

**The reception-first feasibility spike (round-2 H2's "FIRST task") is complete, and the result is decisive: SkillCorner carries a NATIVE completion outcome.** No spatial attribution, no proxy — so the round-2 S1 (attribution-noise bar) and S2 (timebox), and the lakehouse's Consequence-2 release risk, are all **moot**.

**Measured (4 SkillCorner matches via `_loader_pining`, 19,835 events, 2026-06-09):**

| native field | what it is | coverage | non-degenerate? |
|---|---|---|---|
| **`pass_outcome`** (`successful`/`unsuccessful`/`offside`) | pass-completion outcome | goal-kicks **59.6%**, GK-actor passes ~35% | **yes** — goal-kicks ~75% success (42/14), GK-actor ~81% |
| **`received`** | targeted player received the ball | GK-actor passes 64% | yes — 43% received |
| **`player_targeted_id`** (+ name, position) | targeted-receiver **identity** | goal-kicks 95% | native id — no attribution |

**The current `result_id` ignores a native outcome — the precise claim.** On GK-actor passes (n=388): native `received_rate` = 0.43 vs `same_team_next_rate` = 0.84, **agreement = 0.459 ≈ statistical independence**. *Caveat (review N1, self-corrected):* this compares `same_team_next` to **`received`** (the stricter targeted-receipt question), NOT to true completion (`pass_outcome`). So it proves `same_team_next` is a poor **received** proxy; whether it is also a poor **completion** proxy is a separate, unmeasured question (the aggregate `pass_outcome`-success ~81% is actually *close* to `same_team_next` ~84%, so they may agree better at row level — to be measured, plan task 1). What is **certain**: SkillCorner carries an explicit native `pass_outcome` the converter ignores, and using it (vs *any* inference) is strictly better. That is the D-S8 fix; whether the residual-gap inference can keep `same_team_next` is decided by the measurement.

**Label definition — ONE construct, not a 3-way blend (review N1).** SPADL `result_id["success"]` means *the pass reached a teammate*. The corrected SkillCorner `result_id` uses the **single native construct** that matches it, and never mixes in a different or proven-wrong one:
- **Primary — `pass_outcome`:** `successful` → success; `unsuccessful` / `offside` → fail (offside = did not *legally* reach a teammate; **m1: confirm GS native completion treats offside identically — any divergence is a cross-provider seam the D-S9 gate must catch**). This is the SPADL-correct completion. Covers ~all open-play passes + goal-kicks 59.6%.
- **Success-only augmentation — `received==True`:** where `pass_outcome` is absent but the targeted player demonstrably received it, that is a completion → success. **`received==False` is NEVER routed to fail** (review N1): `received` answers a *stricter* question (did the *targeted* player get it) — a pass completed to a *non-targeted* teammate is `received==False` yet a real completion (the spike's 81% `pass_outcome`-success vs 43% `received` proves they are different events). So `received` may only *add* success, never assign failure.
- **Residual gap (neither `pass_outcome` nor `received==True`):** filled by the **construct-correct** "did the ball reach a teammate" inference (the SPADL convention — e.g. via `player_targeted_id` → next-action team, 95% receiver-id coverage on goal-kicks), **flagged** with a new `result_source` provenance column (`native` / `inferred` / `stopgap`) so VAEP and every consumer can see real vs inferred completion — never silently bury quality tiers in the canonical field. **The exact gap mechanism is settled by the Chesterton's-Fence step + a measurement (plan task 1): `same_team_next`-vs-`pass_outcome` row-level agreement.** (My earlier "`same_team_next` is construct-wrong, 0.459 agreement" was measured vs `received`, NOT vs completion — it proves `same_team_next` is a bad *received* proxy, not necessarily a bad *completion* proxy; if it agrees well with `pass_outcome` it is an acceptable flagged stopgap, else use the `player_targeted_id`→next-team reconstruction.)
- **Goal-kicks** now have a native, non-degenerate label → re-measured (the proxy's 0.478 does not bind); model-scored if they clear the floor (m3: use a lower-confidence-bound threshold + report n, given the few-hundred-row sample), else base-rate-served + reported.

**Sourcing the native label — GOLD STANDARD (D-S8, settled): fix the converter `result_id`, accept the VAEP retrain.** The SkillCorner converter routes pass/set-piece `result_id` through the native outcome: **`pass_outcome=='successful'` where present → `received==True` → residual `same_team_next` only where neither is present** (`offside`/`unsuccessful` → fail). This corrects completion for **every** `result_id` consumer — VAEP labels, all completion-dependent features, the lakehouse SPADL — not just xT-GK; one source of truth. The xT-GK completion model then reads `result_id` **uniformly for all providers** — no SkillCorner side-channel column, no per-variant label divergence.
- **VAEP-retrain trigger (accepted):** SkillCorner scores/concedes label distribution shifts (Hyrum). The maintainer prefers one more retrain for the correct long-term architecture; the lakehouse re-materializes SkillCorner VAEP + xT-GK together and waits until ready.
- **Golden updates:** the SkillCorner conversion result distribution changes → committed SkillCorner golden/snapshot fixtures regenerate (the 4.20.1-orientation-golden pattern; the regen is reviewed, not blind).
- **Chesterton's Fence:** verify the converter spec (`2026-05-14-skillcorner-events-converter-design.md`) + `git blame` for *why* `same_team_next` was chosen — it may intentionally cover non-pass/defensive `end_type` rows that `pass_outcome` does not — before replacing it; the fallback chain preserves the prior coverage for those rows.

**Pinned sequence — cross-repo coordinated migration (review N3).** Changing SkillCorner `result_id` changes SkillCorner VAEP (`fct_action_values`), which the lakehouse ingests — so this is a hard cross-repo dependency and the **order is load-bearing** (the D-S9 comparability gate is only meaningful *after* the corrected label/VAEP exists — run on the old proxy data it validates the wrong thing):
1. **Fix SkillCorner `result_id`** (single native construct + `result_source`) **+ regenerate SkillCorner goldens** (reviewed diff). Chesterton's-Fence + the `same_team_next`-vs-`pass_outcome` agreement measurement settle the residual-gap policy here (**plan task 1**).
2. **Retrain VAEP + refit the GK-completion variant** on the corrected label.
3. **Re-measure GS-transfer** on the corrected label (D-S1) → GS-model-reuse vs SkillCorner-specific weights.
4. **Calibration (common-scale vs GS) + AUC gate**, then the **comparability gate** (§2.5/D-S9).
5. **Lakehouse re-materializes** SkillCorner VAEP + xT-GK and **pools** — only after 1–4 pass. ("Lakehouse waits" covers the intent; this pins the ordering.)

## 5. API surface & emitted columns

- `silly_kicks/tracking/_gk_completion.py`: pure **`variant_key_for_provider(source_provider) -> str`** (review C4 — exhaustively CI-testable over all 5 enum values + unknown/None, **no artifact IO**); `GkCompletionModel.from_variant` extended to `{"default"|"gs", "skillcorner"}`; internal variant registry. (`select_variant_for_provider` = the thin `from_variant(variant_key_for_provider(p))` composition — the IO seam, kept separate from the pure mapping.)
- `silly_kicks/spadl/skillcorner.py` (D-S8, the foundational fix): route pass/set-piece **`result_id`** through the single native-completion construct — `pass_outcome=='successful'`→success / `unsuccessful`+`offside`→fail; `received==True`→success **(success-only; `received==False` never→fail, N1)**; residual gap → flagged construct-correct inference. Plus a new **`result_source`** provenance column (`native` / `inferred` / `stopgap`) on the SkillCorner SPADL output so consumers see real-vs-inferred completion. **VAEP-retrain trigger + SkillCorner golden regen.** The xT-GK completion model reads `result_id` uniformly.
- `silly_kicks/tracking/_xt_gk.py`: `compute_xt_gk` / `add_xt_gk` resolve the variant from `frames["source_provider"]` when `completion=None`; `completion=` override unchanged; multi-real-provider → raise (snapshot-excluded).
- New output columns: `xt_gk_completion_variant`, `xt_gk_completion_source`.
- `silly_kicks/tracking/_xt_gk.py` / report: `XtGkReport.spans_multiple_variants` flag (review m-c).
- `scripts/train_gk_completion.py`: `--variant skillcorner` (reads `result_id`); bundles SkillCorner-specific weights **only if** GS-transfer fails the re-measurement (D-S1) — else `skillcorner` resolves to the `gs` artifact.
- Atomic mirror unchanged.

## 6. Decisions for sign-off (each carries a recommendation)

- **D-S0 — scope/release. ✅ SETTLED (user, 2026-06-09): single 4.21.0 with everything.** xT-GK + GS goal-kick coverage + the **native-completion** SkillCorner variant ship together. The round-2 timebox contingency is **moot** — the spike confirmed a native label (§4), so the SkillCorner variant ships construct-correct with no interim proxy and no open-ended attribution branch.
- **D-S1 — per-provider variant *architecture* yes (not a blend); per-provider *weights* now RE-OPENED.** The 0.50 non-transfer was on the *proxy* label (void). On the corrected native label the build **re-measures GS-transfer**: if GS transfers within the gate, `skillcorner` = the GS model (no distinct weights); else SkillCorner-specific weights. The variant-selection seam stays either way (pooling-safe, extensible). **RECOMMEND: re-measure, don't assume.**
- **D-S2 — auto-select via pure `variant_key_for_provider` + `completion=` override + `gs` fallback-with-warning; multi-real-provider raises (snapshot-excluded). RECOMMEND: yes.**
- **D-S3 — calibration is the HARD gate (ECE + reliability-slope), common-scale vs GS; AUC floor secondary (`gs` ≥ 0.84; `skillcorner` ≥ 0.70 on GK-passes, re-measured on the native label; goal-kicks re-measured, model-scored if they clear the floor, else base-rate-served). RECOMMEND: yes (review C1 + lakehouse Consequence 4).**
- **D-S4 — ✅ RESOLVED by the spike (was reception-first fork): SkillCorner ships the NATIVE completion label.** The spike confirmed `pass_outcome`/`received`/`player_targeted_id` are native (§4) — no proxy, no spatial attribution, so S1 (attribution bar) and S2 (timebox) are moot. SkillCorner `xt_gk` is construct-correct (reach-z′), comparable to GS. **RECOMMEND: yes.**
- **D-S5 — metrica stays unsupported (untested) → `gs` + sharper warning. RECOMMEND: yes** (measure before claiming support).
- **D-S6 — model card states the exact native field(s) backing the SkillCorner label (`pass_outcome`/`received`) + coverage. RECOMMEND: yes (Hyrum).**
- **D-S7 — ✅ SETTLED by the lakehouse: `xt_gk` is pooled across providers.** → the comparability gate (§2.5 Consequence 3), common-scale calibration (Consequence 4), and keep-provenance-despite-pooling (Recommendation 3) are now requirements, all folded in. Fork B removed.
- **D-S8 — ✅ SETTLED (gold standard, maintainer 2026-06-09): fix SkillCorner `result_id` to native completion, accept the VAEP retrain.** Corrects completion **to native where native fields exist** (residual rows keep a flagged `stopgap` proxy for VAEP coverage — review m1; the `result_source` tier makes this visible, so the headline must not over-claim "correct everywhere") for every consumer (VAEP + features + lakehouse SPADL); xT-GK reads `result_id` uniformly (no side-channel) and the GK-completion model trains only on `{native, inferred}` (F1). VAEP-retrain trigger + golden regen accepted (the maintainer prefers one more retrain for the correct long-term solution; lakehouse waits). The dedicated-`native_completion`-column band-aid is rejected (it would leave the converter bug in place). Chesterton's-Fence check on `same_team_next` before replacing it.
- **D-S9 — NEW: cross-provider comparability gate blocks pooling until validated (§2.5). RECOMMEND: yes** (lakehouse Consequence 3 — pooling on a quietly-offset scale would mis-rank SkillCorner vs GS keepers).

## 7. Testing (when built)

- **Pure mapping (CI, review C4):** `variant_key_for_provider` over all 5 enum values + `None`/unknown returns the documented key (artifact-free, exhaustive).
- **Variant selection (CI, review m1 — the operationally meaningful guard, lead with this):** `skillcorner` frames → `skillcorner` variant; `{gradientsports, sportec, snapshot}` → `gs`; unknown/`metrica` → `gs`+warning; `completion=` override beats auto-selection; two distinct real providers → raise; lone `snapshot` (or `snapshot`+one real) → no raise.
- **SkillCorner `result_id` native-completion fix (CI, D-S8 + N1):** synthetic fixtures asserting the **single construct** — `pass_outcome=='successful'`→success, `'unsuccessful'`/`'offside'`→fail; `received==True`→success; **a `received==False` row that is a real completion (e.g. completed to a non-targeted teammate) is NOT marked fail** (the N1 regression — guards against re-introducing the blend); residual-gap rows carry `result_source ∈ {native, inferred, stopgap}` correctly. **SkillCorner conversion goldens regenerated** (reviewed diff) + CHANGELOG/ADR **VAEP-retrain-trigger** note. A guard documents `result_id` for non-pass/defensive rows is unchanged (Chesterton's-Fence scope check).
- **Residual-gap policy measurement (plan task 1, owner/public-run):** `same_team_next`-vs-`pass_outcome` row-level agreement decides whether `same_team_next` is an acceptable flagged stopgap or the `player_targeted_id`→next-team reconstruction is needed; reported, not assumed.
- **GS-transfer re-measurement on the corrected label (train/owner-run, D-S1):** report GS→SkillCorner AUC/calibration on the *native* label; decide GS-model-reuse vs SkillCorner-specific weights (the 0.50 proxy-label number is explicitly superseded).
- **Calibration gate, common-scale (train script, owner/public-run, review C1 + lakehouse C4):** ECE + reliability-slope within tolerance is a hard pass/fail, ahead of the AUC floor, checked against the GS variant's reliability target (not per-variant in isolation); reported for both variants.
- **SC AUC floor on the NATIVE label (train script, owner/public-run, review m2):** GK-pass held-out AUC ≥ 0.70 on `native_completion`; goal-kicks re-measured on the native label — model-scored if they clear the floor, else base-rate-served (the path taken is reported).
- **Cross-provider comparability gate (train/owner-run, D-S9 + lakehouse C3):** reception-SC vs native-GS `xt_gk` distributions on matched distance/zone bands show no provider-systematic offset beyond tolerance; a failure blocks pooling (or mandates a documented re-scaling).
- **Provenance (CI):** `xt_gk_completion_variant` / `xt_gk_completion_source` populated correctly (SC goal-kick → `model` if it cleared the floor else `base_rate`; SC GK-pass → `model`); `XtGkReport` per-variant counts == column value-counts.
- **Consumer contract — runtime signal (CI, review H1 + m-c):** `XtGkReport.spans_multiple_variants` is `True` when scored inputs span >1 `xt_gk_completion_variant`, `False` within a single variant. (Preferred over a doc-presence lint; the no-pooling-without-comparability rule also present in the column docs + ADR-024.)
- **Train==serve parity (CI):** SkillCorner variant builds features through the same `extract_gk_completion_features` / `_resolve_gk_geometry` path (atol=1e-9), inherited from the GS gate.
- **Construct-distinctness (CI, review m1 — framed honestly):** `gs` and `skillcorner` are distinct artifacts (different SHA256). Note: this guards "no one collapsed them to one file," **not** the construct finding itself — the variant-selection test above is the meaningful guard.
- **Inherited xT-GK gates:** construct-invariant, nan-safety, id-dtype, dup-action_id, atomic-mirror parity, provenance-skip idempotence.

## 8. Effort

≈ Tier-4: SkillCorner converter `result_id` native-completion fix (~30 LOC) + **SkillCorner golden regen + VAEP-retrain coordination (lakehouse)** + variant registry + pure `variant_key_for_provider` (~60 LOC) + selection wiring (~30 LOC) + 2 provenance columns + `spans_multiple_variants` flag + calibration/common-scale + comparability gates in the train path + `--variant skillcorner` + bundled SC weights **iff** GS-transfer fails the re-measurement + consumer-contract docs + tests. The reception feasibility spike is **done** (§4, native label confirmed). No new runtime dependency (pure-numpy logistic serve).

---

## 9. Sources
- Probe A (GS `default` → fixed SkillCorner, N=460): `reference_skillcorner_converter_data_bugs` §(3), task `bd8oifsyo` — 2026-06-09.
- Probe B (SkillCorner-trained OOF CV, N=547): task `bmrfuf35j` — 2026-06-09.
- IDSSE transfer (0.824): goal-kick-coverage spec §3.3 R2.
- sportec-direct native completion: `silly_kicks/spadl/sportec.py:816–858` (play_evaluation override), verified 2026-06-09.
- SkillCorner reception fields: `silly_kicks/spadl/skillcorner.py:217–224` (`player_targeted_x/y_reception`), verified 2026-06-09.
- **Native-completion spike** (4 SC matches, 19,835 events): `pass_outcome`/`received`/`player_targeted_id` coverage + the proxy-vs-native agreement 0.459 — tasks `byeabn0gf` (raw column dump) + `b02aabwxa` (GK-row coverage + proxy contrast), 2026-06-09.
- snapshot frames-only converter: `silly_kicks/tracking/_snapshot.py` (all-`snapshot` per match), verified 2026-06-09.
- Provider enum: `silly_kicks/tracking/schema.py:29,69`.
- Construct/bugfix context: `reference_skillcorner_converter_data_bugs`, `2026-06-08-xt-gk-goalkick-coverage-design.md`, ADR-024.

---

## 10. Review round 1 (parallel critic, 2026-06-09) — resolutions

- **H1 (proxy `p` ≠ the `p` RAV consumes; consumer-side mismatch unaddressed) — ACCEPTED.** Added §2.5: hard no-cross-variant-pooling contract + provider-local qualification in column docs/report/ADR-024; `mean(xt_gk)` valid only within a variant; lakehouse pooled-use surfaced for sign-off (D-S7). Exec-summary fact #4.
- **H2 (proxy-now bakes a future silent semantic shift on a published column) — ACCEPTED.** D-S4 resolved to **reception-first**: a feasibility spike (§4) leads the SkillCorner work; fork (A) ship the construct-correct reception variant, (B) ship the proxy *explicitly interim/provider-local*. No deferred semantic-shift debt; both branches inside 4.21.0.
- **C1 (calibration must be a hard gate for a multiplicatively-consumed probability) — ACCEPTED.** §3.2: ECE + reliability-slope is the hard gate, ahead of the AUC floor, for any RAV-probability variant (both `gs` and `skillcorner`). Exec-summary fact #5.
- **C2 (sportec → gs rests on kloppy-path evidence; the enum can't distinguish sportec-direct) — VERIFIED + RESOLVED.** Checked `spadl/sportec.py:855–858`: the **direct** converter derives native completion from DFL `play_evaluation` (`unsuccessful → fail`, non-degenerate ~71%) — same construct as the kloppy gateway, so `sportec → gs` is sound. Also surfaced + documented the event-provider-vs-tracking-provider distinction: auto-selection assumes they match; cross-provider stacks use `completion=` (§2.2). `snapshot → gs` justified (completion rides the actions/event provider).
- **C3 (mixed-provider raise may false-positive on synthetic snapshot frames) — VERIFIED + RESOLVED.** `_snapshot.py` builds an all-`snapshot` frames table per match (never appended to a provider) — no real path mixes them. The guard now excludes `snapshot` from the distinct-real-provider uniqueness check (§2.2), cheap insurance.
- **C4 (`select_variant_for_provider` not pure — loads artifacts) — ACCEPTED.** Split into pure `variant_key_for_provider(provider) -> str` (CI-testable, artifact-free) + the `from_variant(key)` IO seam (§5, §7). The mapping is the logic locked in CI.
- **m1 (no-transfer SHA guard oversold) — ACCEPTED.** §7 leads with the variant-selection test as the meaningful guard; the SHA-distinctness check is framed as "didn't collapse to one file," not a construct guarantee.
- **m2 (SC gate is owner/public-run, not a PR CI gate) — ACCEPTED.** Stated in §3.2 + §7.
- **m3 (`is_throw_in +0.204` is inert) — ACCEPTED.** Flagged in §1 (near-empty positive class, not signal — same as the GS plan).
- **Closing question (what does the lakehouse do with SC `xt_gk`?) — SURFACED (D-S7).** If pooled-per-player across providers, H1 is the headline use case → reception-first or an explicit provider-segregated consumption contract on the lakehouse side.

## 11. Review round 2 (parallel critic, 2026-06-09) — resolutions

Verdict: ship-quality pending two scoping refinements; architecture accepted; round-1 resolutions spot-checked + verified-in-source.

- **S1 (un-validated spatial-attribution reception could masquerade as construct-correct) — ACCEPTED.** §4 step 3 + the fork now require, when reception is built via attribution (no native receiver-id), that **attribution accuracy be validated past a pre-registered bar** (≥ ~0.9 vs ground truth) *before* fork (A) is allowed to claim construct-correct; below the bar it is fork (B) (interim/provider-local). Test added (§7). Closes the loophole where a noisy attribution-reception label escapes the §2.5 qualification.
- **S2 (front-loaded open-ended spike on the 4.21.0 critical path) — ACCEPTED.** §4 step 4 + D-S0: the cheap native-field check stays on the critical path; the expensive attribution+validation branch is **timeboxed** — overrun → ship fork (B) in 4.21.0, reception becomes a bounded follow-up. A SkillCorner variant always ships; the spike's depth never holds the release hostage. Reception-first intent preserved (we still lead with the cheap check).
- **m-a (state why auto-selection keys on the tracking provider) — ACCEPTED.** §2.2: actions carry no provider column → `frames["source_provider"]` is the only signal, a proxy for the event/label provider, exact only single-stack, `completion=` handles mismatch.
- **m-b (calibration ≠ correctness on the proxy branch) — ACCEPTED.** §3.2: "calibrated" = well-scaled to the proxy outcome (retention), not to RAV's reach-z′; the no-pooling + provider-local qualification stay essential on the proxy branch even with calibration green.
- **m-c (consumer contract is advisory; prefer a runtime signal) — ACCEPTED.** §2.5 + §7: `XtGkReport.spans_multiple_variants` flag makes a mixed-variant aggregation observable, preferred over a doc-lint; real enforcement stays on the lakehouse side (D-S7).
- **D-S7 (lakehouse pooled-per-player use) — NOW SETTLED (see §12).** The lakehouse confirmed `xt_gk` IS pooled across providers → fork (A) construct-correct reception is **required**. The spike (§12) confirmed the native label exists, so the requirement is met, and the comparability + common-scale gates are added.

## 12. Spike result + lakehouse decision (2026-06-09) — resolutions

**Reception feasibility spike (round-2 H2's "FIRST task") — DONE, decisive (4 SC matches, 19,835 events):**
- **Native completion confirmed.** `pass_outcome` (`successful`/`unsuccessful`) covers goal-kicks 59.6% (~75% success, non-degenerate) + GK-actor passes ~35%; `received` covers GK-actor passes 64%; `player_targeted_id` is a native receiver id (95% on goal-kicks). **No spatial attribution needed → S1 (attribution bar) + S2 (timebox) MOOT.**
- **`same_team_next` is at minimum a poor *received* proxy** (agreement with native `received` = 0.459 ≈ independence, n=388). *Self-correction (rev-4 N1):* that compares to `received`, NOT to true completion (`pass_outcome`) — so it does **not** by itself prove `same_team_next` is a bad *completion* label (aggregate `pass_outcome` ~81% ≈ `same_team_next` ~84%; row-level agreement is plan task 1). What is certain: SkillCorner carries an explicit native `pass_outcome` the converter ignored, and using it beats any inference. Fork B (a proxy-labeled pooled column) is off the table regardless.
- **Resolution:** SkillCorner ships the **native** completion label (D-S4 resolved to "native," superseding "reception-first fork"); §4 rewritten; §2.1/§2.3/§3.1/§3.2/§3.3 updated to native; goal-kicks re-measured on the native label (no longer auto-excluded).

**Lakehouse decision — `xt_gk` pooled across providers in `fct_action_context` (D-S7 settled):**
- **Fork B off the table** (pooling + no partition → an interim proxy would realize H1 in production). Resolved by the native label.
- **D-S9 comparability gate (NEW)** — pooling requires reception-SC vs native-GS distribution comparability (matched bands, no systematic offset) before pooling; folded into §2.5 + §3.2 + §7.
- **Common-scale calibration (lakehouse C4)** — the C1 gate is checked against the GS reliability target, not per-variant (§3.2).
- **Keep provenance despite pooling (lakehouse rec 3)** — `xt_gk_completion_variant` + `XtGkReport.spans_multiple_variants` ship as the only post-hoc detection/repair seam (§2.5).

**D-S8 — ✅ SETTLED (gold standard, maintainer 2026-06-09): fix SkillCorner `result_id` to native completion, accept the VAEP retrain.** Supersedes the dedicated-column band-aid. Rationale: already retraining VAEP for the 4.20.x bug fixes; one more retrain buys the correct long-term architecture — SkillCorner completion correct for *every* consumer, xT-GK reads `result_id` uniformly. Scope/time not a constraint (lakehouse waits). Bundled into 4.21.0 as the foundational change. **Reopens D-S1** — per-provider *weights* now contingent on re-measuring GS-transfer on the corrected label (the 0.50 was wrong-label).

## 13. Review round 4 (parallel critic, result_id gold-standard rev, 2026-06-09) — resolutions

Verdict: architecture is the cleanest yet; the one fix is the label-definition blend now feeding the canonical `result_id`.

- **N1 (corrected `result_id` was a 3-construct blend feeding VAEP) — ACCEPTED, the core fix.** `result_id` = **single construct, `pass_outcome` primary** (SPADL "reached a teammate"); `received==True` is **success-only** (`received==False` NEVER→fail — it answers the stricter targeted-receipt question; the 81%-`pass_outcome` vs 43%-`received` gap proves they differ); residual gap → **flagged** construct-correct inference via a new **`result_source`** column (`native`/`inferred`/`stopgap`), never a silent proxy. The exact gap mechanism is a Chesterton's-Fence + `same_team_next`-vs-`pass_outcome` measurement (plan task 1). §4/§5/§2.5/§7 + D-S8 updated. **Self-correction:** my "0.459 ⇒ same_team_next is construct-wrong" conflated `received` with completion — corrected throughout (§1/§4/§12).
- **N2 (internal contradictions from layered revisions) — FIXED.** Exec-fact #1 ("no native pass-outcome" → false) rewritten; §2.5 stale proxy-branch bullet removed (both variants now one construct); recommendation + §2.1 moved off the "pass_outcome/received" blend to `result_id`/`pass_outcome`-primary.
- **N3 (cross-repo retrain→gate sequencing) — PINNED.** §4: explicit 5-step order (fix `result_id`+goldens → retrain VAEP+refit → re-measure GS-transfer → calibration+comparability gate → lakehouse re-materialize+pool); the D-S9 gate is only meaningful *after* the corrected label exists.
- **N4 (comparability gate can fail but pooling is locked) — CONCRETE.** §2.5: tolerance→pool; stable modest offset→**validated affine re-scaling** (the chosen path, since pooling is committed); large/uncorrectable→**escalate** (SC doesn't pool until understood); + a **minimum overlap-band n** (thin SC-short/GS-long goal-kick overlap → under-powered bands reported, not silently passed).
- **m1 (offside) — NOTED:** `offside`→fail, with a build check that GS native completion treats offside identically (a cross-provider seam the D-S9 gate must catch). §4.
- **m2 (label tier ≠ inference path) — ADDED:** §2.3 — `result_source` weights training only; at inference any geometry-present row is model-scored; base-rate-served is geometry-missing rows only.
- **m3 (goal-kick small-sample gate) — ADDED:** §3.2/§4 — lower-confidence-bound threshold + report n for the goal-kick model-vs-base-rate switch.
