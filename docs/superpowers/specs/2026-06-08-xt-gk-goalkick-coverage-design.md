# xT-GK goal-kick coverage — coordinate derivation + completion model (design spec)

## Executive summary (for review)

Running xT-GK on real Gradient Sports data (owner-tier, WC2022) exposed **two coupled gaps** that make the composite `xt_gk` NaN for most real goal-kicks:

1. **Missing origin coordinate.** ~**63–67% of GS goal-kicks have a NaN *start* coordinate** (the destination is usually present). xT-GK can't value a distribution with an unknown origin → NaN.
2. **Out-of-distribution completion model.** The `accessible-space` xC that RAV currently uses for `P(success)` is built on **open-play passes** and resolves for only **~31% of real goal-kicks** (the aerial / ball-placement regime defeats it).

Either gap alone nulls the composite; together they cap real goal-kick coverage at a small fraction. This spec closes both, **honestly tagged**, so the composite is defined for ~100% of in-scope goal-kicks and every value carries machine-readable provenance.

**Part A — goal-kick coordinate derivation (scoped to xT-GK).** Derive the missing origin via a **conditional origin** (native where present → tracking-GK position where available → empirical native-start median → goal-area rule point `(5.5, 34)`), used **identically at train and serve** (§3.3, M2). The **derived coordinates feed the xT-GK valuation internally**; the shared `actions` frame is **never mutated** (a converter-level change would alter `start_x/end_x` for every downstream consumer — VAEP, the xT fit, all features — a large Hyrum/retrain blast radius and its own ADR). **Confidence (measured, not asserted):** the imputed origin is *not* value-neutral — it propagates through the completion model's length/forwardness features into RAV (the dominant term). Measured: swapping native→prior origin moves `P(success)` by mean 3.2 pp (median 1.3, p90 8.5, max 29), implied |ΔRAV| ~4–5% typical / ~12% p90 with a long tail. So imputed-origin rows are **usable but lower-confidence**, and ship **tagged** (`xt_gk_origin_source`) and **gated separately** (§6, M5) — never as silently-equal to native-origin values.

**Part B — fitted GK-distribution completion model (replaces the OOD xC for RAV).** Completion is **observable** (SPADL `result_id`), so we *fit* `P(success | geometry)` rather than guess. A measured model bake-off (below) selects **logistic regression**. This becomes RAV's primary `P(success)` and **drops `[das]` from a hard xT-GK requirement to optional**.

**Provenance for outside inspection.** Per-row `*_source` enum columns ride in the output wide table (`xt_gk_origin_source`, `xt_gk_dest_source`), the established `is_goalkeeper_source` / `speed_source` idiom — queryable/filterable downstream — plus an optional aggregate `XtGkReport` for pipeline QA.

**Decisions needing sign-off** are in §7 (each carries a recommendation; D2 is already settled by measurement).

---

**Date:** 2026-06-08 · **Author:** Karsten (with Claude) · **Status:** **Approved for implementation** — all §7 decisions confirmed by the maintainer 2026-06-08.
**Context:** folds into the in-flight xT-GK release (silly-kicks 4.21.0, ADR-024). xT-GK itself is implemented + green (3888 tests); this closes the real-data RAV/coverage gaps before it ships. Amends ADR-024 ("RAV P(success) = get_xc" → fitted completion model; `[das]` now optional).
**Attribution:** within Jeffrey Eyestone's delegated *"derive P(success) from the library's pass-completion machinery."* A purpose-built fitted GK-completion model **is** that machinery — more faithful than open-play xC here. FYI to Jeffrey; not a blocker.

---

## 1. Evidence (real GS data, owner-tier)

Goal-kick coordinate NaN pattern (43 goal-kicks, 3 matches; representative):

| | share |
|---|---|
| NaN **start** coord | **67%** |
| NaN end coord | 12% |
| NaN both | 12% |
| linked to a tracking frame | 100% |
| …ball tracked at that frame | 28% |
| next event has a start coord | 88% |
| native start cluster | x: median 8.8 (p10 5.4, p90 26.9); y: median 32.2 (central) |

xC coverage (RAV `P(success)`): finite for **31% (4/13)** of real goal-kicks; the resolved values are plausible — the failure is coverage.

Three real provider-robustness bugs were also found + fixed en route (NaN coords in `_grid_value`, unlinked rows crashing `get_xc`, heterogeneous-frame batch assertion) — already merged + unit-tested.

## 2. Part A — goal-kick coordinate derivation (scoped to xT-GK)

The missing coordinate is overwhelmingly the **origin**. NOTE (review H1): the origin is value-light for `base`/`dzv` (origin-xT ≈ 0.001–0.005, flat) but **value-bearing for RAV** — it drives the completion model's `length`/`forwardness` features (§3.2). So the derivation must be measured, train/serve-consistent, and confidence-tagged — not waved off.

**Conditional origin (review M3), in confidence order, used identically at train + serve:**
1. **Native** start, where present (~42% of goal-kicks-with-destination).
2. **Tracking-GK position** at the linked frame — **clamped to plausibility (review R3): only when the GK is tracked within the goal/penalty area (x ≤ 16.5 m).** Measured: the acting-team GK is tracked at 100% of goal-kick frames, but **48% of the time it is off-position (> 16.5 m up the pitch)** — using it then yields a wildly wrong `length` (the value-bearing feature, H1), worse than the median it's meant to outrank. Off-position GK falls through to tier 3. (The ball is even less reliable — tracked at only ~28% of goal-kick frames — so it is not a tier.)
3. **Empirical native-start median** `≈ (8.8, 32)` (preserves some origin realism vs a single point).
4. **Goal-area rule point** `(5.5, 34)` (6-yard-box centre; goal-kicks are taken from the goal area by Law) — terminal fallback.

`xt_gk_origin_source` records which tier fired. **Destination:** native (88%) → next-event start (`end ≈ next action's start`, standard SPADL) → tracking-ball; `xt_gk_dest_source` records it. If neither origin nor destination resolves → row stays NaN (honest).

**The derived coordinates FEED the valuation (review m7/m8):** `_resolve_gk_geometry` runs **before** the coord gate and supplies the `sx/sy/ex/ey` that drive `base/progress/rav/dzv/pressure` — it is not a tags-only annotation. The shared `actions` frame is never mutated (derivation is internal to xT-GK).

**Scope decision (D-A1):** derivation lives **inside xT-GK** via the private `_resolve_gk_geometry`; a general converter-/`spadl`-level coordinate-enrichment feature (codebase-wide, retrain-triggering) is a **separate future ADR** (logged in TODO).

**Applicability:** goal-kicks (type 22) get the goal-area prior (rule-justified). Keeper passes/throws have a *variable* open-play origin — no fixed prior; they rely on native coords or tracking-ball/GK, else unresolved.

## 3. Part B — GK-distribution completion model (RAV `P(success)`)

### 3.1 Model bake-off (measured, match-grouped 5-fold CV, GS data)

| model | N=100 (goal-kicks, 16 matches) | N=1814 (all GK distributions, 40 matches) |
|---|---|---|
| base-rate Brier | 0.182 | 0.174 |
| logistic (length only) | AUC 0.791 · Brier 0.175 | AUC 0.803 · Brier 0.134 |
| **logistic (all feats)** | **AUC 0.804 · Brier 0.172** | AUC 0.876 · Brier 0.119 |
| XGBoost (all feats) | AUC 0.767 · Brier 0.178 | **AUC 0.893 · Brier 0.110** |

Small data → logistic wins (XGBoost overfits); realistic data → XGBoost wins **modestly** (+1.7 AUC pts; ~16% more Brier-skill-over-base-rate). Both crush the baseline and are well-calibrated.

**Train==serve correction (review M2, measured).** The table above was fit on the **coord-present** subset (the rows that had a native origin). The production model serves on imputed-origin rows too, so the honest number is the fit built **through the same `_resolve_gk_geometry` imputation** (train==serve). Measured on 40 GS matches (N=605 goal-kicks-with-destination, 58% imputed origin): **logistic AUC 0.831** / Brier 0.145; XGBoost AUC 0.858 / Brier 0.141; coord-present-only logistic 0.839 — i.e. the train/serve skew is real but **small (~0.008 AUC)** once features are built through the imputation pipeline. **The honest headline is AUC ≈ 0.83, not 0.876.** Training MUST build features through `_resolve_gk_geometry` (a train==serve feature-parity test gates this, §6).

**Decision D2 → logistic regression.** The XGBoost edge is modest and does not justify pulling **`[xgboost]` onto the RAV path** + the full ADR-011 weights-PR/booster-JSON/HPO ceremony. Logistic captures most of the skill (**honest train==serve AUC ≈ 0.83**, review R4 — *not* the coord-present-only 0.876, superseded by M2), serves **pure-numpy** (`sigmoid(Xβ)`, **zero new runtime dependency** — replacing `[das]` with *nothing*), and ships **interpretable coefficients** ("completion drops X% per 10 m"). XGBoost (≈ 0.86) is the **reserved ADR-011 upgrade** if held-out calibration later demands it.

### 3.2 Features (geometry, leakage-free, state-anchored)
Pass length `‖z′−z‖`, forwardness `Δx/‖·‖`, lateral offset `|Δy|`, destination x, destination-defender density (`receiver_zone_density`), aerial/type flag (goal-kick vs keeper-pass vs throw). Exact set is a review point (§7 D3). Label = `result_id == success` (confirm provider semantics, §7 D5).

### 3.3 Fit / provenance / serialization
**Bundled `default` is GS-trained (review R1).** D2 and every AUC/calibration number were measured on the **GS** slice of `_loader_pining` only — so the shipped default is the **GS-trained** model that matches that measurement, *not* a multi-provider blend whose quality was never measured. A **multi-provider default is a measured follow-up** (re-run the bake-off + per-provider label-semantics check on SkillCorner/IDSSE first — see m6/D5: the degenerate-label guard catches an all-one-class column but **not** a subtler cross-provider difference in what `result_id==success` means for a goal-kick, which would silently blend two label definitions). Fit offline, **features built through the SAME `_resolve_gk_geometry` imputation used at serve** (review M2) — training never drops imputed-origin rows, so train==serve geometry is identical (a feature-parity test gates this, §6). Match-grouped CV. Ship a small **JSON coefficient artifact** — coefficients + **explicit `feature_names` + order** + standardization stats + SHA256, **no pickle** (the ADR-012-style envelope, review m9, so a reindex bug can't silently misalign coefficients) — as bundled `default`, **caller-overridable** via `completion=`. `scripts/train_gk_completion.py` (held-out AUC/Brier/reliability + permutation importance). Pure-numpy inference. Leakage discipline as xS/xCross/xT (disjoint fit, state-anchored features).

**Label semantics + degenerate-label guard (review m6).** Label = `result_id == success`; **verify per provider (GS/SkillCorner/IDSSE) before training** — if a provider marks all goal-kicks `success` (no contest modelling) the label is degenerate and the model collapses to base rate. `prepare_gk_completion_training_data` **fails loud** on a degenerate label distribution (e.g. min-class fraction below a floor) rather than silently shipping a base-rate model. Cross-provider OOD (fit on GS, served on SkillCorner/IDSSE) is accepted given `completion=` override + provenance tags — stated, not hidden.

**Self-sufficiency (review M4 + R5).** The model returns a **per-type calibrated base rate** when features are missing/degenerate — conditioned on the aerial/type flag (goal-kick ≈ 50/50 long, keeper-pass, throw-in ≈ near-certain have very different completion rates; a single global base rate would mis-value the fallback path). It never delegates to `get_xc`. So xT-GK's RAV path references no `_das` at all (§3.4); `[das]` does not silently re-enter as a soft fallback.

### 3.4 Integration
RAV consumes `p = completion.predict(features(a))` for in-scope rows. **`get_xc` / `_require_das()` are fully removed from xT-GK's RAV path** (review M4 — verified Chesterton's Fence: within `compute_xt_gk`, `[das]`/`get_xc` feeds *only* RAV; `base`/`dzv`/`pev`/`pressure` use the xT grid + `pressure_on_actor` geometric/velocity kernels, none touch accessible-space — so removal is clean). The composite no longer references `[das]`. ADR-024 amendment paragraph. Construct gates unchanged. The coverage claim is **split** (review M5, §6): native-origin calibration is the verified signal; imputed-origin coverage is informational + drift-alarmed — "finite" is never the green criterion.

## 4. Provenance return mechanism (outside inspection)

- **Primary — per-row enum columns** on the `add_*` / `compute_*` output wide table (the `is_goalkeeper_source` idiom):
  - `xt_gk_origin_source` ∈ `{native, goalkick_prior, tracking_ball, unresolved}`
  - `xt_gk_dest_source` ∈ `{native, next_event, tracking_ball, unresolved}`
  - (linkage provenance `frame_id` / `time_offset_seconds` / `link_quality_score` already emitted)
  The value is returned (coverage ↑) **and** tagged, so the lakehouse can audit/filter/weight (`WHERE xt_gk_origin_source = 'native'`).
- **Continuous confidence (review R7).** A categorical tier alone lets a *careful* consumer filter, but the lakehouse will likely consume `xt_gk` raw (Hyrum), and the measured imputed-origin tail reaches |ΔP(success)| = 29 pp. Emit a continuous **`xt_gk_origin_confidence` ∈ [0,1]** (native = 1.0; tracking-GK-in-area ≈ 0.7; empirical-median ≈ 0.4; rule-point ≈ 0.2 — values pinned at implementation) so downstream can **threshold or down-weight** rather than only hard-filter by tier. **Documented recommended filter:** for player-evaluation use, restrict to `xt_gk_origin_confidence ≥ 0.7` (native + in-area tracking); use the full set only for population aggregates. Reuses the `link_quality_score` idiom.
- **Secondary (optional) — aggregate `XtGkReport`** (counts/fractions per source) from `compute_*`, mirroring `ConversionReport`/`LinkReport`, for pipeline QA + drift alarms. (Equivalently derivable via `GROUP BY` — so a convenience, not load-bearing.)
- **Not on the VAEP `xt_gk_xfns` path** — that emits model features per gamestate slot, not provenance, by design.

## 5. API surface & emitted columns

- `silly_kicks/tracking/_gk_completion.py` (new): `GkCompletionModel` (`predict`, `from_variant("default")`, `from_json`/`to_json`), `extract_gk_completion_features`, `prepare_gk_completion_training_data`, `compute_gk_completion`.
- `silly_kicks/tracking/_xt_gk.py`: `_resolve_gk_geometry(actions, frames, links)` private helper → derived `(origin, dest, origin_source, dest_source)`; `compute_xt_gk`/`add_xt_gk` gain `completion: GkCompletionModel | None = None` (default → bundled); `[das]` no longer required.
- New output columns: existing 6 (`xt_gk_base/pev/rav/dzv/pressure/xt_gk`) + `xt_gk_origin_source`, `xt_gk_dest_source`.
- `scripts/train_gk_completion.py` (`[train]`); bundled `default` JSON artifact committed.

## 6. Testing

- **Model quality** (train script): AUC / Brier / reliability vs base-rate + distance-only; pure-numpy-serve == sklearn-fit parity; JSON round-trip + SHA + `feature_names`-order alignment (m9); leakage/no-self-fit.
- **Train==serve feature parity (review M2 + R6, CI):** the durable assertion is that **the same `extract_gk_completion_features` code path** constructs features at fit and serve (incl. imputed-origin rows via `_resolve_gk_geometry`); the numeric check is `np.allclose(..., atol=1e-9)` (NOT "byte-identical" — pandas-fit vs numpy-serve can differ by a ULP and would flake). Plus **missing-feature parity (review R2):** a row with a NaN tracking feature is handled identically at train and serve.
- **Degenerate-label guard (review m6, CI):** `prepare_gk_completion_training_data` raises on an all-one-class label distribution (synthetic fixture) — never silently a base-rate model.
- **Split coverage gate (review M5):**
  - **(a) native-origin calibration — the green criterion:** on a held-out **native-origin** set, AUC ≥ baseline + reliability within tolerance (real signal; must be right where we can verify). Owner-run on real data.
  - **(b) imputed-origin coverage — informational + drift alarm:** report the imputed fraction + the imputed-origin `P(success)` distribution; alarm on drift. **Finiteness is NOT a pass criterion** (house rule: never silently substitute data).
- **Self-sufficiency (review M4 + R8, CI):** the test must **execute with accessible-space actually absent** — `monkeypatch` the `accessible_space` import to raise `ImportError` (or an uninstalled-`[das]` CI lane) — and assert xT-GK still produces RAV/composite. A symbol-not-called assertion is a no-op where `[das]` happens to be installed; this forces the real independence.
- **Coordinate derivation:** conditional-origin tiers fire in confidence order (synthetic native / tracking-GK / NaN-origin fixtures); `*_source` tags correct; **derived coords drive the compute** (imputed-origin row yields a non-NaN composite, m7/m8); never mutates input `actions`.
- **Provenance:** every in-scope scored row gets a non-null `*_source`; enum-value contract; `XtGkReport` counts == column value-counts.
- **Inherited xT-GK gates:** construct-invariant, nan-safety, id-dtype, dup-action_id, atomic-mirror parity, provenance-skip idempotence.
- **CI vs owner (review m10):** owner GS e2e is not a PR gate (GS not in CI); the three risks the owner smoke can't gate — imputed-origin path, degenerate-label guard, train==serve parity — are covered by **synthetic CI fixtures** above.

## 7. Decisions — ALL CONFIRMED (user, 2026-06-08)

- **D2 — ✅ logistic regression** (measured, §3.1); XGBoost reserved as the ADR-011 upgrade.
- **D-A1 — ✅ scoped *inside* xT-GK now, designed for promotion, general version tracked as a follow-up.** The derivation helper (`_resolve_gk_geometry`) is built pure / provenance-emitting / xT-GK-decoupled so a future **general coordinate-enrichment feature** (converter- or `spadl`-level, codebase-wide, with its own ADR + tripwire + a *coordinated* model retrain) can lift it out. Reason for not generalizing now: a codebase-wide goal-kick-coordinate change is a Hyrum/retrain trigger for every model (VAEP, xT, calibration) and must not be bundled into the xT-GK release. Follow-up logged in TODO.
- **D-A2 — ✅ origin imputation = CONDITIONAL** (native → tracking-GK → empirical median ≈ (8.8, 32) → rule point (5.5, 34)), not a single fixed point (review M3). The earlier "choice barely affects value" is **corrected by measurement** (review H1): the origin is value-light for base/dzv but drives RAV via the completion length feature (|ΔRAV| ~4–5% typ / ~12% p90). Hence the conditional tiers + train==serve consistency + lower-confidence tagging.
- **D1 — ✅ drop `[das]` as a hard xT-GK requirement** (get_xc stays for open-play DAS).
- **D3 — ✅ feature set as §3.2; train on ALL GK distributions** (the model's real domain).
- **D4 — ✅ bundle a `default` (fit on pining) + caller-overridable** (`completion=`), like xS/xCross.
- **D5 — ✅ `result_id == success` is the completion label; verify provider semantics** (GS/SkillCorner/IDSSE) at implementation before training.
- **D6 — ✅ fold into 4.21.0** (xT-GK ships with real goal-kick coverage).

## 8. Effort

≈ Tier-5: `_gk_completion.py` (~150–250 LOC) + `_resolve_gk_geometry` helper (~80 LOC, conditional-origin) + train script + bundled artifact + RAV wiring + provenance columns + report + ADR-024 amendment + tests. The xT-GK robustness fixes + σ-scan are already done.

---

## 9. Review round 1 — parallel critic (2026-06-08), resolutions (measured, not asserted)

- **H1 (origin DOES move RAV) — ACCEPTED, measured.** The completion `length`/`forwardness` features depend on the origin, so the imputed origin propagates into RAV (the dominant term). Measured (40 GS matches, coord-present rows, native→prior swap): |ΔP(success)| mean 3.2 pp (p90 8.5, max 29); implied |ΔRAV| ~4–5% typ / ~12% p90, long tail. Resolution: the "barely moves the value" claim is removed; imputed-origin rows are usable but **lower-confidence**, tagged + split-gated.
- **M2 (train/serve geometry skew) — ACCEPTED, measured + resolved.** Features now built through `_resolve_gk_geometry` at train (no row-dropping); honest train==serve AUC ≈ 0.83 (vs the optimistic coord-present-only 0.876). Parity test gates it (§6).
- **M3 (single-point prior collapses variance) — ACCEPTED.** Conditional origin (§2, D-A2).
- **M4 (get_xc must be fully removed) — ACCEPTED.** Self-sufficient model; no `_das` on the RAV path; CI assertion (§3.4, §6).
- **M5 (coverage gate rewards finiteness, not correctness) — ACCEPTED.** Split gate: native-origin calibration is the green criterion; imputed coverage is informational + drift-alarmed (§6).
- **m6 (label degeneracy) / m7–m8 (derived coords feed compute) / m9 (feature_names envelope) / m10 (CI fixtures for the owner-ungated risks)** — ACCEPTED, folded into §2/§3.3/§6.
- **Verified-for-us by the reviewer:** `[das]` removal is Chesterton-clear (RAV-only consumer); "pining corpus" == the `_loader_pining` set (term unified, §3.3).

### Review round 2 (parallel critic, 2026-06-08), resolutions

- **R1 (default trained on an unmeasured corpus) — ACCEPTED.** Ship the **GS-trained default** that matches the measurement; multi-provider is a measured follow-up (per-provider `result_id==success` semantics verified first). §3.3.
- **R2 (serve-time feature availability) — MEASURED, holds.** `receiver_zone_density` is **finite at serve for 92%** of real goal-kicks (frame has opponents 100%; the feature is computed at the goal-kick frame for the destination zone, so it does not need a destination-time frame). Keep it; the 8% (NaN destination) rides the missing-feature path; a missing-feature train==serve parity test added (§6).
- **R3 (tier-2 off-position) — MEASURED, clamped.** GK tracked off-position (> 16.5 m) for **48%** of goal-kicks → tier-2 gated on in-goal-area plausibility, else fall to tier-3 (§2).
- **R4 (AUC citation) — FIXED:** all justifications cite the honest ≈ 0.83; 0.876 explicitly labelled coord-present-only/superseded (§3.2, §3.3, §9).
- **R5 (per-type fallback) — ACCEPTED:** self-sufficiency base rate conditioned on the type flag (§3.3).
- **R6 (float parity) — ACCEPTED:** shared-`extract_gk_completion_features`-code-path assertion + `atol=1e-9`, not "byte-identical" (§6).
- **R7 (continuous confidence) — ACCEPTED:** `xt_gk_origin_confidence ∈ [0,1]` + documented recommended downstream filter (≥0.7 for player-eval) (§4).
- **R8 (no-`_das` test mechanism) — ACCEPTED:** runs with accessible-space monkeypatched-absent, not symbol-not-called (§6).
- **R9 (forward pointer, out of scope):** when the lakehouse adopts `xt_gk` as a per-player GK evaluative metric it triggers downstream AI-governance obligations (model card / `AI_GOVERNANCE.md`). Nothing for silly-kicks; **flag on the consumer PR**.

---

### Sources
- Owner-data verification (read-only `_loader_pining`, GS WC2022): coordinate-NaN probe, σ-scan, goal-kick xC smoke, logistic-vs-XGBoost bake-off — all 2026-06-08.
- xT-GK design + consent trail: `docs/superpowers/specs/2026-06-07-xt-gk-design.md`, ADR-024.
- Precedent (trained-light, pure-numpy serve, bundled+overridable, leakage-safe): xS (TF-16) / xCross (TF-17), ADR-011; provenance idiom: `is_goalkeeper_source`, `LinkReport`/`ConversionReport`.
