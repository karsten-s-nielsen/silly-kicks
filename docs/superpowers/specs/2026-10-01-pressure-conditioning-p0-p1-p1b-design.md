# silly-kicks pressure conditioning — P0 + P1 + P1b design (APPROVED; parked for TF-66)

**Date:** 2026-10-01 · **Status:** **APPROVED** (Rev 3; 3 `/review-spec` rounds, ≥3 passes, unanimous APPROVE at round 3 — D1-SPEC-06 + fixture-tier + §8.1 all resolved). Committed to the repo as the **PARKED** design for **TF-66** (not built; no branch; no plan yet — the plan is written at cycle-start and travels with the impl). §7 owner decisions RESOLVED (2026-10-01: P1b denominator = per opponent-possession primary + per-minute secondary; `link_zones`/`bekkers_pi` glossary backfill = in-scope for P1); the ONLY remaining gate is sequencing after the GK-observability cycle (see TODO TF-66). Provenance: research handoff → multi-pass `/review-spec`. Drafted by Claude Opus 4.8 (`claude-opus-4-8`). ⚠ `file:line` anchored at `main@517c200`; re-anchor at plan time.
**Target repo:** `silly-kicks`, on `main` `517c200` (verified clean). Every `file:line` is anchored on `main` `517c200` — re-anchor at plan time.
**Research basis:** `docs/research/pressure-conditioning/RECOMMENDATION.md` (SOTA survey + the P0–P4 roadmap + the §8 citation table; the "RECOMMENDATION §N" references below point into it). Third-party paper PDFs are NOT in-repo (copyright); citations live in `NOTICE` + RECOMMENDATION §8.
**Review history:** matured over 3 independent `/review-spec` rounds (≥3 passes per round; reviewer model id + skill version recorded per pass), unanimous APPROVE at round 3. The per-round findings + resolutions are in the Rev-2 / Rev-3 change logs below.

## Rev-2 change log (what moved + why)

- **B1 (BLOCKING, F1/D1-SPEC-01):** the import-allowlist guard does NOT cover `gi.py`. Rev 2 specs a new AST import-boundary gate. §2.
- **B2 (BLOCKING, D1-SPEC-02+03 — the analytics 3/3 set MISSED this; caught by pass 1 + verified here):** re-graining `compute_pressing_kpis` breaks the `team_metrics` multi-producer merge. Rev 2 makes P1b **augment** (new output + new contract family), not a re-grain. §4.
- **F2:** "Bekkers 2024" is 34 sites / 15 files; Rev 2 scopes the fix to live source+tests + living ADRs, leaves historical docs. §0.
- **F3:** honest-NaN vocab gains `raw_pressure_na`; precedence + partial-cohort rule stated. §3.
- **F4 — RESOLVED on a real public GI slice (match 1886347, SkillCorner Open Data):** `block` is NOT a raw column and NOT derived from `organised_defense`; it is a token-subset of `team_out_of_possession_phase_type`. §2, §7.
- **F5:** the z-score re-expression is NEW code (not `_pressure_levels` reuse); Rev 2 homes it in a new `tracking/_pressure_conditioning.py`. §3, §8.
- CONSIDERs folded throughout.

## Rev-3 change log (round-2 re-reviews)

- **D1-SPEC-06 (SHOULD FIX, re-review #1 — #2 missed it):** `block` is a token-subset of `phase_out_of_possession` (F4), so a `phase_out × block` cohort is **collinear/degenerate**. Rev 3 pins the P1 cohort to **`phase_in_possession × block`** and forbids/warns `phase_out × block`. §3.
- **Fixture-tier (SHOULD FIX, re-review #2 — #1 missed it):** the GI schema is **tier-variable** — the public WC-finals 34-col `dynamic_events` lacks `organised_defense`/`n_defensive_lines`/`game_state_id` present in the 294-col A-League slice. Rev 3 requires the P0 "keys surface" test to use a **full-GI (294-col) fixture carrying all keys** (else Part-G vacuous), with a separate omitting-tier fixture for the all-`<NA>` path. §2.
- **Naming/anchor (CONSIDERs):** serialization methods are `to_meta`/`from_meta` (not `to_dict`/`from_dict`); `_pressure_levels` quantile at `:65/:76` (not `:77`). §3.
- **C7 (CONSIDER):** named the min-cohort-n off-by-one boundary test (`n_min` vs `n_min-1`) + the CI gate (`ci.yml` runs pytest over `tests/`). §3/§5.
- **§8.1 CLOSED (post-APPROVE fold, public slice 1886347):** possession-keying verified clean; P0 reduces `game_state_id`/`n_defensive_lines` from the `player_possession` anchor row (they drift ~5%/~2.4% within a possession); the SPADL-action→possession 1:1 mapping deferred to plan (needs the converter). §2/§8.

---

## 0. Housekeeping fact-check (step 1) — verified on `main` `517c200`

**ONLY actionable finding: the "Bekkers 2024" year.** Everything else the handoff listed is already correct.

| Handoff/RECOMMENDATION claim | Verdict | Evidence |
|---|---|---|
| NOTICE cites arXiv `2501.00712` → `2501.04712` | **already `2501.04712` + "Bekkers, J. (2025)"** | `NOTICE:262-263`; `git log -S "2501.00712" -- NOTICE` → no commits (never wrong); correct since `14d8634` (3.2.0/PR-S25) |
| `reaction_time` (τr) mis-attributed to the paper | **already attributed to UnravelSports** | `pressure.py:71` (`pressing_intensity.py L120`) |
| active-pressing `speed_threshold` has no paper value → redirect | **the paper DOES give 2 m/s; attribution correct** | Bekkers v2 §3.1 / Fig-2 caption (arXiv:2501.04712 v2; full text in the external paper corpus, cited in NOTICE + RECOMMENDATION §8); `pressure.py:76-77`. The RECOMMENDATION §6 "redirect to UnravelSports" is **wrong** — overturned. |
| `pressure.py` docstring "Bekkers 2024" (arXiv is 2025) | **TRUE** | see count below |

**Scope of the "Bekkers 2024" fix (corrected count — Rev 1 said 9):** **34 occurrences across 15 files.** Split:
- **Fix now (live source + tests, 9 sites / 5 files):** `pressure.py:6,66`; `_kernels.py:633,712`; `features.py:1076,1134`; `atomic/tracking/features.py:1248`; `test_pressure_bekkers.py:1,274`. Docs/comments only, zero behaviour change, inconsistent with NOTICE (already 2025).
- **Do NOT rewrite (historical docs, 25 sites / 10 files):** `CHANGELOG.md` (×3), dated `docs/superpowers/specs/*` (~11), `plans/*` (~9), `ADR-004`, `ADR-063`. History lives in git; rewriting shipped CHANGELOG/dated-spec/plan text for a citation-year nit violates the no-rewrite-history convention.
- **Owner decision:** the 2 living ADRs (ADR-004:21, ADR-063:242) — fix as living docs, or leave as historical? Surface, don't decide.

**Separate pre-existing repo bug (not the pressure cycle):** `gi.py:6-7` docstring claims it is "pinned by `tests/providers/test_appearances_import_allowlist.py`" — **false** (that test globs `*/appearances.py`, never `gi.py`). Owner repo-fix; folded into B1's gate work below.

All housekeeping edits are docs/comments, **gated on Karsten's explicit per-commit approval**.

---

## 1. Context

Three instantaneous, phase-blind `pressure_on_actor` primitives (`andrienko_oval` default, `link_zones`, `bekkers_pi`), aggregated pooled across phase/block. The gap is **conditioning**, above the primitive dispatch. The keys ride the SkillCorner GI feed, unused (**verified unused**: the raw field names appear nowhere in `silly_kicks/providers/` or `tests/providers/`). P0 plumbs them; P1 re-expresses pressure within a phase×block cohort; P1b adds de-pooled team-pressing KPIs.

**Anti-patterns closed:** P1 → **E6 + E2** (all three primitives); P1b → **E2 + E5**. **E7 NOT closed** (the sibling GK-observability cycle's axis; §7). **Validation:** current 14 pressure test files are **construct-only**; P1/P1b ship construct + face validity, predictive/robustness **deferred to P4**; the conditioned output must **not** inherit an "empirically validated" claim.

---

## 2. P0 — ingest the GI conditioning keys (plumbing; SkillCorner-only)

**Seam:** `silly_kicks/providers/skillcorner/gi.py` — a pure shaping function (pandas; `id_compat` added by P0 for id routing; **never** `silly_kicks.tracking`). Raw loading stays scripts-side.

**F4 resolution (verified on real public GI slice, match 1886347 / SkillCorner Open Data, 294 cols):**
- Raw columns that EXIST: `team_in_possession_phase_type` (+ `_id`), `team_out_of_possession_phase_type` (+ `_id`), `game_state_id` (+ `game_state`), `n_defensive_lines`, `organised_defense`, `defensive_structure`.
- **There is NO raw `block` column.** Block height is the **`high_block`/`medium_block`/`low_block` token-subset of `team_out_of_possession_phase_type`** (full out-of-poss vocab: `chaotic, defending_direct, defending_quick_break, defending_set_play, defending_transition, disruption, high_block, low_block, medium_block`). In-poss vocab: `build_up, chaotic, create, direct, disruption, finish, quick_break, set_play, transition`.
- So `block` is a **trivial vocabulary VIEW** of `phase_out_of_possession` — **not raw, not `organised_defense`-derived** (the reviewer's F4 hypothesis is wrong). P0 stays plumbing + a documented map; `organised_defense` is NOT needed and stays out of scope with no contradiction.

**`parse_conditioning_keys(gi_events, *, game_id) -> DataFrame`**, one row per possession event, declared columns:

| column | source | dtype | notes |
|---|---|---|---|
| `game_id` | `str(game_id)` | str | |
| `period_id` | `period` | `Int64` | |
| `decision_id` | `associated_player_possession_event_id` | str | the join key (SAME grain `parse_passing_options` calls `decision_id`; one name, not two) |
| `team_id` | `team_id` | id via `id_compat` | |
| `phase_in_possession` | `team_in_possession_phase_type` | `category` | STATIC low-card (ADR-103 F1a) |
| `phase_out_of_possession` | `team_out_of_possession_phase_type` | `category` | STATIC low-card |
| `block` | **map `phase_out_of_possession`: `{high_block,medium_block,low_block} → {high,medium,low}`, else `<NA>`** | `category` | a VIEW, not a raw field; a `*_SOURCE_VALUES` tuple documents the 3 tokens |
| `gi_game_state_id` | `game_state_id` | `Int64` | non-colliding name (glossary `game_state` = scoreline, false friend) |
| `n_defensive_lines` | `n_defensive_lines` | `Int64` | |

**Grain / key-reduction (verified §8.1 on public slice 1886347, 944 possessions):** the keys ride the GI events one-to-many (a possession has 1-12 child events: passing_option/on_ball_engagement/off_ball_run point back via `associated_player_possession_event_id`; the `player_possession` row IS the possession and carries a NULL `associated_player_possession_event_id` — it is the ANCHOR). `team_in_possession_phase_type` + `team_out_of_possession_phase_type` are **single-valued per possession (0/944 multi)** → `phase_in`/`phase_out`/`block` assign unambiguously. But `game_state_id` (~5%, 49/944) and `n_defensive_lines` (~2.4%, 23/944) **drift within a possession** → **source them from the `player_possession` anchor row** (the canonical possession-level value), NOT modal/any; `n_defensive_lines` is partial-null (~49-55% populated) → honest-`<NA>`. `team_id` is per-event (688/944 multi) → take the possessing team from the anchor, never assume possession-constancy.

**Contracts:** fail-closed (missing field → `<NA>`, column present; absent field → all-`<NA>`, reportable, no crash); no value coercion beyond the block map; keys reduced from the `player_possession` anchor (above); SkillCorner-scoped; closed vocab documented.

**B1 — the import-boundary guard (BLOCKING fix).** `gi.py` is currently **unswept** (`test_appearances_import_allowlist.py:50` globs `*/appearances.py`; its 2nd gate requires `keeper_identity` `:114` — neither fits `gi.py`). P0 adds a **new** test `tests/providers/test_skillcorner_shaping_import_boundary.py` that AST-sweeps **every `providers/skillcorner/*.py` shaping module** (gi.py + any future one) for the banned prefixes (`silly_kicks.tracking`, …) **without** the `keeper_identity` requirement, plus a **planted-violation meta-test** (mirror `:140-161`) proving the gate actually fires. The P0 guard is THIS test — not the appearances sweep. Also corrects `gi.py:6-7`'s false docstring.

**⚠ GI schema is TIER-VARIABLE (re-review #2):** the 294-col A-League slice carries all keys, but the public WC-finals 34-col `dynamic_events` lacks `organised_defense`/`n_defensive_lines`/`game_state_id`. P0's fail-closed all-`<NA>` contract handles an omitting tier, but a "keys surface" test run on such a tier passes **vacuously** (Part G).

**Tests:** empty/no-GI-event → empty declared frame; the **keys-surface test MUST use a full-GI (294-col) fixture that ships every conditioning key** (else vacuous) → keys surface at the `decision_id` grain and join 1:1 to option/possession rows; a **separate omitting-tier fixture** → all-`<NA>`, no crash (the honest-NaN path); the `block` map covers all 3 tokens + NA elsewhere; the new import-boundary gate + its planted-violation meta-test pass.

---

## 3. P1 — aggregation-conditioning layer (E6 + E2)

**Pattern:** aggregation-conditioning (Bischofberger 2026). Compute the raw primitive uniformly, re-express it (z-score) within a phase×block cohort. Additive; no primitive re-fit; no new labels; not in the default VAEP feature space. Closes **E6 + E2** for all three primitives.

**F5 — home it in a NEW module `silly_kicks/tracking/_pressure_conditioning.py`, not a `_pressure_levels` extension.** `xtgk/_pressure_levels.py` is tercile-quantize + raise-on-empty (`np.quantile [1/3,2/3]` `:65/:76`; raise `:63,:69`) with `to_meta`/`from_meta` serializing only `band_cutpoints` (`:120/:133`) — it does NOT z-score, does NOT serialize a z-score, and its band axis is hard-coded binary, so cohort-keying is genuinely **new code**, not "reuse almost unchanged" (Rev 1 over-claimed). The new module consumes `PressureLevels` only if a tercile view is also wanted; the z-score path is its own:
- **Cohort axis = `phase_in_possession × block` (D1-SPEC-06, re-review #1).** `block` is a token-subset of `phase_out_of_possession` (F4), so `phase_out × block` is **collinear/degenerate** — block adds no information, "closes E2" would be hollow, and the independent-block non-vacuity test (below) would pass trivially. The orthogonal pairing is the attacker's phase (`phase_in`, the E6 axis) × the defensive block height (`block`, the E2 axis). The param dataclass permits `phase_in × block` (primary), `phase_in` alone, `phase_out` alone, or `block` alone; it **forbids/warns `phase_out × block`** (degenerate).
- `fit(pressure, *, cohort) -> per-cohort (μ, σ)` keyed by the cohort label, raising on NOTHING (honest-NaN, F3).
- `transform` → z-score of raw pressure within its cohort.
- serialization (its own `to_meta`/`from_meta`) that round-trips the per-cohort (μ, σ) — a real z-score branch, not `band_cutpoints`.
- a named **min-cohort-n off-by-one boundary test** (`n_min` passes, `n_min-1` → `cohort_too_small`) + the CI gate is `ci.yml`'s pytest-over-`tests/` (C7).
- xtgk↔tracking import direction verified legal (the module lives in `tracking/`; it does not import `xtgk` unless the tercile view is opted in — confirm at build).

**Add-a-metric surface:** flavor-suffixed column `pressure_on_actor__<method>__z_<cohort>` (never overwrites raw); frozen param dataclass (cohort axis, min-cohort-n floor); `feature_glossary` entry for the new flavor **AND backfill the pre-existing gap** (`link_zones`/`bekkers_pi` lack flavor entries — **in-scope for P1, Karsten 2026-10-01**: close the coverage-gate gap while in the glossary file); NOTICE (Bischofberger 2026); **NOT in `pressure_default_xfns` NOR `atomic_pressure_default_xfns` (`atomic/tracking/features.py:1304`)** — opt-in only; `add_*` purity (ADR-033) + the ADR-078 **canonical** call convention (not "frame-aware" — that's ADR-020) if it ships as an `add_*`.

**F3 — honest-NaN vocabulary (closed data contract; adding a token later is a breaking migration, so get it complete now):** `pressure_conditioning_source ∈ {z_scored, raw_pressure_na, cohort_too_small, cohort_unavailable}`.
- `raw_pressure_na` — **NEW** (F3): the raw primitive is `<NA>` for THIS action inside a present, adequately-sized cohort (e.g. `bekkers_pi` with missing velocity / a detection-censored position). Distinct from `cohort_unavailable` (the whole cohort axis absent, e.g. non-SkillCorner).
- **Precedence:** raw-`<NA>` (→ `raw_pressure_na`) is checked FIRST, before cohort-size.
- **Partial-cohort rule:** μ/σ computed over the non-`<NA>` subset of the cohort; the min-n floor applies to that subset; below it → `cohort_too_small`.
- Never a bare 0/1.

**Validation (construct + face only; predictive deferred to P4):**
- Construct: z ~ mean-0/unit-scale within cohort (assert); monotone (higher raw → higher z within cohort).
- **Non-vacuity on BOTH axes independently (F4/D1-SPEC-04 — E2 and E6 are separate):** a **phase** plant/scramble (inject a per-phase shift → recovered; scramble phase labels → ~0 differential) AND an **independent block** plant/scramble. A block_conditional variant must not ship vacuous while phase tests stay green.
- Face: reproduce Markou-2024's "signal flat when pooled, visible per-phase" on public data only (SkillCorner Open Data A-League MIT / Sportec IDSSE CC-BY), labelled face-validity, NOT predictive (E4).
- NOT claimed: predictive/robustness/≥2-channel/cross-league.

---

## 4. P1b — de-pooled team-pressing KPIs (E2 + E5) — AUGMENT, not re-grain

**B2 (BLOCKING fix).** `compute_pressing_kpis` (`_pressing.py:39`) is ONE producer of the `team_metrics` mart: `_compute.py:88-92` outer-merges `pressing ⋈ progression ⋈ buildup` on `(game_id, team_id)` (`TEAM_KPI_KEYS`), registered as the single `team_metrics` family in `metric_contracts.py:74`. **There is no "pressing family."** Re-graining `compute_pressing_kpis` to phase×block would break that merge and the `team_metrics` contract. Rev 1 called P1b both "additive" and "a contract change" — contradictory. **Resolution: AUGMENT.**
- **Leave `compute_pressing_kpis` + the whole-match `team_metrics` mart + its merge UNCHANGED** (byte-identical).
- Add a **separate** de-pooled producer (e.g. `compute_pressing_kpis_by_cohort(... ) -> DataFrame` at grain `(game_id, team_id, phase, block)`) as a **NEW output with its OWN `metric_contracts` family** + grain keys (ADR-098; `tests/test_metric_contracts.py` reds if unregistered). Not merged into the `(game_id,team_id)` mart.
- **E5 per-opportunity normalisation — denominator DECIDED (Karsten 2026-10-01):** raw counts (`recoveries`, `counterpress_regains` — `_PRESSING_COLUMNS`, `_pressing.py:32-35,106`) normalised by an opportunity denominator. **PRIMARY = per opponent-possession (count)** (the unit of pressing opportunity; interpretable as the fraction of opponent possessions regained; matches the PPDA/exPress convention), **SECONDARY = per minute of opponent-possession** (duration-robustness cross-check). Source opportunities from the existing `spells` possession definition (don't invent). Raw counts retained; denominator-0 cohort → `<NA>` (not inf/0).
- Depends on P0 (phase/block keys). Must not ship before P0.

Additive (a new output, not a re-grain) → genuinely no existing byte change, minor bump, no `team_metrics` re-materialise. Construct (per-cohort rows reconcile to the pooled total where that identity holds; rate finite/bounded); face; predictive deferred.

---

## 5. Cross-cutting

Additive, no retrain, no re-materialise; minor `_version.py` bump (ADR-079) + CHANGELOG; one feature branch off `main`, no worktree; public-data-only for external numbers; fail-closed/honest-NaN everywhere. The `*_default_xfns` exclusions (both `pressure_default_xfns` and `atomic_pressure_default_xfns`) keep the default feature space byte-identical.

## 6. Out of scope (future cycles)

P2 (pitch-control-at-ball Spearman/Fernández + Narizuka seconds-contract; verify Spearman-2018 units), P3 (value-linked pressing model), P4 (full validation ladder), Peters rest-defence (not a pressure input). **Annotate RECOMMENDATION §6's overturned speed-threshold error before P2 inherits it.**

## 7. Flags for Karsten (surface, don't decide)

1. **Two blockers, and the analytics 3/3 review missed one.** B1 (gi.py guard) was unanimous; **B2 (P1b re-grain breaks the `team_metrics` merge) was caught only by the earlier single pass + verified here against `_compute.py:88-92`** — "unanimous 3/3" was unanimous on the verdict, not coverage. Re-review should confirm B2.
2. **Sequencing vs TF-64 (GK detection-gate wiring).** ADR-109's `detected_mask` primitive shipped (code-only); **TF-64** is the wiring follow-up — see the TF-64 TODO row + `docs/superpowers/specs/2026-09-27-detection-primitive-design.md` §8. **TF-64 is itself gated after TF-58**, so the provider/tracking chain is **TF-58 → TF-64 → TF-66**. All touch `providers/`+`tracking/`; one branch per cycle, order them. **E7** is TF-64's axis — P1/P1b condition pressure but don't gate on detection; `raw_pressure_na` (F3) is where a detection-censored `bekkers_pi` lands, and the two should eventually compose (conditioned pressure also observability-gated).
3. **P1b opportunity denominator** (§4) — ✅ DECIDED 2026-10-01: per opponent-possession (primary) + per-minute (secondary).
4. **"Bekkers 2024" year** (§0) — ✅ RESOLVED in `949730e`: live source/tests + the 2 living ADRs fixed; historical docs left as git history.
5. **Glossary backfill** of `link_zones`/`bekkers_pi` flavors (§3) — ✅ DECIDED 2026-10-01: in-scope for P1.
6. **gi.py:6-7 docstring** false import-allowlist claim — ✅ RESOLVED in `949730e`.

**Only remaining gate: #2 (GK-observability sequencing).** All other §7 items are decided/resolved.

## 8. Open verifications (before a plan)

- **CLOSED (§8.1, verified on public slice 1886347):** possession-keying is clean — `phase_in`/`phase_out` single-valued per possession (0/944); `game_state_id`/`n_defensive_lines` drift (~5%/~2.4%) → reduced from the `player_possession` anchor (§2). **Residual, defer to plan:** the final SPADL-action→`associated_player_possession_event_id` 1:1 mapping needs the converter output (full tracking load); GI-side keying is clean, so low-risk.
- The new `tracking/_pressure_conditioning.py` import direction (must not create a tracking→xtgk edge unless the tercile view is opted in).
- `to_meta`/`from_meta` z-score round-trip design for the new module (F5).
- (CLOSED: GI key existence/vocab + `block` raw-vs-derived — verified on match 1886347, §2.)

## 9. Re-review instructions

Independent multi-pass `/review-spec` (≥3 passes, agreement reported, reviewer model id + skill version recorded). Confirm B2 specifically (the prior 3/3 missed it). Verify the F4 slice resolution, the F3 vocab completeness + precedence, the B1 gate design (does the planted-violation meta-test actually fire?), and the §8 import direction. Probe omissions (Part F).
