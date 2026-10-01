# Pressure metrics in silly-kicks — state of the art and a gold-standard roadmap

**Date:** 2026-09-30 · **Status:** research recommendation, nothing built. Investigation only; scope and any
build remain Karsten's call. Read-only against silly-kicks `main` (a parked TF-58 branch is held
separately — not touched).

## TL;DR

The critique that started this ("Andrienko lumps everything into one category") is real but points at the
**wrong layer**. silly-kicks already ships three interchangeable pressure primitives; Andrienko is only the
default. The weakness is not the oval — it is that **none of the three conditions on context**, and the
aggregations on top pool across phase and block. So there are two independent axes:

- **Axis A — the instantaneous primitive.** A modest upgrade exists (we already have the better one,
  `bekkers_pi`); a genuinely different improvement is a pitch-control-at-ball primitive that aggregates all
  defenders coherently. Secondary priority.
- **Axis B — conditioning (the real E6/E2 fix).** Making pressure phase/block/state-dependent. This is a
  **still-open gap in the published literature** (even the newest value-linked model, exPress 2025,
  conditions only on the last three actions; block-type conditioning is essentially absent). The fix sits
  **above** the primitive dispatch and applies to all three methods at once. **This is where the
  gold-standard, differentiated maturation is.**

**Recommended near-term:** ingest the conditioning keys SkillCorner already ships → add a method-agnostic
aggregation-conditioning layer → fix the pooled team-pressing KPIs. **Flagship (longer):** a value-linked,
phase-conditioned pressing model validated on the full ladder — which would be the first published work to
close E6/E2 rather than narrate it.

---

## 1. Current silly-kicks state (main)

- **Three primitives** in `silly_kicks/tracking/pressure.py` (+ `_kernels.py`): `andrienko_oval` (default;
  geometric oval, no velocity), `link_zones` (zone occupancy), `bekkers_pi` (Bekkers 2025 probabilistic
  time-to-intercept; needs velocity; honest-NaN on velocity-less SB360). Multi-flavor dispatch, frozen
  param dataclass per method (ADR-005 §8). All three are **instantaneous, per-dyad, phase-blind**.
- **Pooling exposure:** `pressure_default_xfns` = one global Andrienko applied to every phase (E6). Team
  pressing KPIs (`team_metrics/_pressing.py`, `compute_pressing_kpis`) group only by `(game_id, team_id)`,
  whole-match — no phase, block, or per-opportunity split (E2 + E5). The causal layer already treats
  `pressure_on_actor__bekkers_pi` as a **confounder** alongside score/time — i.e. the code already concedes
  pressure varies with context, but only to control for it, never to report it conditionally.
- **Reusable conditioning machinery:** `xtgk/_pressure_levels.py` — `PressureLevels(mode="global"|
  "zone_conditional")` buckets pressure into terciles, already fits per-band cutpoints. Its only axis today
  is pitch zone; extending `Mode` with `phase_conditional` / `block_conditional` is the natural template.
- ⭐ **The conditioning keys already exist and are unused.** The raw SkillCorner GI schema carries
  `game_state_id`, `team_in_possession_phase_type`, `team_out_of_possession_phase_type`,
  `organised_defense`, `defensive_structure`, `n_defensive_lines` on every event — but the provider parser
  (`providers/skillcorner/gi.py`) ingests none of them. Conditioning is a **plumbing PR, not new detection
  logic**. (`game_state` in the glossary today is scoreline only — a false friend.)
- ⚠ **Current validation is construct-only.** The 14 pressure test files check physical invariants, ranges,
  and golden-master parity (Bekkers vs unravelsports) + cross-provider numerical consistency — **not**
  outcome or ground-truth. A conditioned variant must not inherit an "empirically validated" claim.
  (Pre-existing gap: only `andrienko_oval` has a glossary flavor entry; `link_zones`/`bekkers_pi` don't.)

## 2. Axis A — the instantaneous primitive

Ranked by physical grounding and, separately, by validation:

| Primitive | Uses velocity | Aggregates all defenders | Outcome-validated | Note |
|---|---|---|---|---|
| Andrienko oval (2017) | no | Σ, ad hoc | construct only | our default; the E6 anchor; expert-set params |
| Link zones (2016) | no | Σ + saturation | qualitative | legacy |
| **Bekkers TTI (2025)** | **yes** | 1−∏(1−p), independence | none (methods paper) | **already the best primitive we ship**; naive independence assumption |
| **Pitch-control-at-ball** (**Spearman 2018** primary; Fernández–Bornn 2018 fallback) | yes | **yes, coherently incl. GK** | Spearman: OBSO-validated (OBSO→OBSO PCC 0.60, team goals PCC 0.76). Fernández–Bornn: **none** (qualitative video only, "no ground truth") | reads pressure as 1−attacking-control at the carrier; only non-single-dyad family. PPCF integral is compute-heavier |
| Narizuka min-arrival-time (2026, arXiv:2606.09452) | yes | — | **yes, 306 matches** (progression↓, turnover↑, zone-stratified) | **2 fixed params (α=1.0, Vmax=10) vs TTI's 5**; outputs seconds, not [0,1] |

**Verdict (full text read 2026-09-30):** we do not need to "replace Andrienko" — `bekkers_pi` already is the
better primitive, and Andrienko stays as the **velocity-free floor** that works on providers without
velocities (that is *why* it is the default, not an oversight). Two worthwhile additions:
(1) a **pitch-control-at-ball** primitive on the **Spearman family** — `bekkers_pi` and Spearman-2018's PPCF
are *siblings*, both direct extensions of the same **Spearman et al. 2017** pass-probability paper (not
Bekkers-on-Spearman-2018), so PPCF is a low-conceptual-distance upgrade sharing the TTI+logistic kernel,
combining every defender + keeper coherently via the field integral instead of an ad hoc sum/max. Prefer
Spearman over Fernández–Bornn (same math family, the only one outcome-validated, Bayesian-fit params);
Fernández–Bornn only as a lighter fallback if PPCF's per-player time-integral proves too costly.
(2) **Narizuka min-arrival-time** as a 4th primitive — **2 fixed constants vs the TTI family's 5**, its own
dual-channel outcome validation; ⚠ it outputs a *time in seconds*, not [0,1], so folding it into
`pressure_on_actor`'s contract (report seconds vs define a time→pressure transform) is a real design
decision, not plumbing. ⚠ No newer primitive is proven better than Bekkers head-to-head — "better" = more
parsimonious / more directly outcome-validated, not benchmarked-superior.

## 3. Axis B — conditioning (the real gap, and the differentiated work)

Confirmed **open in the literature**: no paper bakes phase into the primitive formula, and block-type
conditioning is almost entirely absent (one throwaway mention across the 11-paper corpus). Even exPress
(2025, the newest value-linked pressing model) conditions only on the preceding three actions. The corpus
does show four method-agnostic patterns, each sitting **above** the primitive dispatch (so they apply
uniformly to `andrienko_oval` / `link_zones` / `bekkers_pi` / any future primitive):

1. **Sample-restriction** — compute the same primitive but only within one phase. Precedent: Markou 2024
   (build-up only), Dash et al. 2025 (defensive transition only). ⭐ Markou's key result: conditioning
   *surfaces a signal pooling hides* — defensive-line height is only significant once you restrict to
   build-up. Direct evidence that per-phase reporting changes conclusions.
2. **Parameter-conditioning of the formula by context** — same shape, context-dependent constants.
   Precedent is **zonal, not phase**: Herold/Forcher's goal-distance-shrinking oval; Merckx's own-box
   threshold relaxation. Extending this to phase×block is novel but needs calibration → **uncalibrated-
   parameter risk** (you would trade E6 for an unvalidated parameter unless you fit per bucket first).
3. **Aggregation-conditioning** — compute the raw primitive uniformly, then re-express it (z-score or
   percentile) against a phase/block/state-matched baseline. Precedent: Bischofberger et al. 2026
   (role/structure-conditioned baselines). **Reuses `_pressure_levels` almost unchanged** (add a
   `phase_conditional` mode), needs no re-fit of the primitive and no new labels, is additive (doesn't touch
   the default VAEP feature space), and closes **E6 and E2 for all three primitives at once.** Best
   value-for-effort.
4. **Value-linked model** — feed phase/block/state + the raw primitive into a model predicting an outcome,
   letting context interact non-additively. Precedent: exPress XGBoost (target P(regain ≤ 5s)),
   VPEP/Merckx risk-reward. Closes E6/E2 **and** the validation gap; fits SkillCorner *better than its own
   source data* (continuous 22-player tracking vs StatsBomb-360 partial frames; the P(regain) target needs
   no shot events, sidestepping SkillCorner's missing-shots gap). Biggest effort — needs a label + a model.

## 4. Validation — the gold standard the corpus implies

Chain all four rungs: **construct → face → predictive (held-out) → robustness**; require **≥2 outcome
channels** (turnover *and* threat/value, not turnover-only — E3); **stratify by phase and block**; and check
**cross-league / cross-season stability**. No paper in the corpus clears more than two of these four
simultaneously (Merckx clears the most: construct + predictive + temporal split-half + external correlation).
Our current pressure suite sits at rung one. Maturing the *validation* is as much of the gold-standard move
as any new metric — and it is exactly Part B of the shared research-discipline standard (see the
`research-discipline` skill / `docs/research/README.md`).

## 5. Recommended roadmap (staged; each item names the anti-pattern it closes)

| # | Item | Closes | Effort | Notes |
|---|---|---|---|---|
| P0 | **Ingest GI conditioning keys** into the SkillCorner provider (`phase_type` in/out, block, `game_state_id`, `n_defensive_lines`). | enabler | low | Plumbing only; the fields are already in the feed. Prerequisite for everything below. |
| P1 | **Aggregation-conditioning layer** above the primitive dispatch (extend `_pressure_levels` `Mode` with `phase_conditional`/`block_conditional`; a `pressure_on_actor` context re-expression). Report pressure z-scored within phase×block. | **E6 + E2** for all 3 primitives | medium | Additive; no calibration; no new labels; reuses existing machinery. The core "phase-dependent pressure" deliverable. |
| P1b | **Fix pooled team-pressing KPIs** — group `compute_pressing_kpis` by phase×block and/or per opportunity, not whole-match. | **E2 + E5** | low-med | Needs P0's keys. |
| P2 | **Pitch-control-at-ball primitive** (5th `method=`, **Spearman family**) + **Narizuka min-arrival-time** (4th; both confirmed on full read). | axis-A upgrade | medium | Independent of B. Pitch-control = only family aggregating all defenders + GK (Spearman primary/outcome-validated; Fernández fallback). ⚠ Narizuka outputs seconds → contract decision. ⚠ PPCF compute-heavier. |
| P3 | **Value-linked pressing model** (exPress-style; target P(regain ≤ Ns); phase/block/state as features). | E6/E2 **+ E3 validation** | high | Flagship; first published work to *close* the gap. Needs a label + model + the full validation ladder. |
| P4 | **Validation ladder** applied to whatever ships (construct→face→predictive→robustness; ≥2 channels; per-phase; cross-league). | E3/E4 + rigor | med, ongoing | Matures the suite past construct-only. |

Near-term high-value, low-risk: **P0 → P1 (+P1b)**. Flagship research: **P3**. P2 is an independent primitive add.

## 6. Caveats and verify-before-cite

- Keep Andrienko as the velocity-free default; don't flip the default to Bekkers globally (SB360 has no
  velocities → honest-NaN). Route to the stronger primitive only where velocity exists.
- Prefer aggregation-conditioning (pattern 3) over parameter-conditioning (pattern 2) until per-bucket
  calibration exists — otherwise you replace a pooling flaw with an unvalidated parameter.
- **Citation due-diligence done + all five FULL-TEXT read 2026-09-30 — see §8.** Bekkers digit CONFIRMED
  wrong in NOTICE (it cites `2501.00712`, an unrelated language-models paper; correct is `2501.04712`) —
  a one-char fix for the silly-kicks cycle (don't edit the parked TF-58 checkout).
- ⚠ **Bekkers param provenance (actionable for the silly-kicks cycle):** the paper states σ=0.45 and T=1.5s
  (our `bekkers_pi` matches), but gives **no numeric value** for reaction time τr or the active-pressing
  speed threshold — those come from the UnravelSports reference code, not the arXiv text. Audit
  `bekkers_pi`'s τr + speed-threshold constants; if either is attributed to "Bekkers 2025", redirect the
  citation to the UnravelSports source. Also: `pressure.py` docstring says "Bekkers 2024" — the arXiv is 2025.
- ⚠ **Before hard-coding Spearman-2018 kinematics:** the accel / max-speed values extracted as "5 m/s and
  7 m/s²" look unit-transposed vs the standard Shaw convention (max-speed ≈5 m/s, accel ≈7 m/s²) — verify
  against the rendered PDF page. And the shared Spearman et al. 2017 pass-probability paper is listed with
  inconsistent author order across the citing papers — use the arXiv/proceedings order if citing directly.
- **Peters (rest defence) is NOT a pressure-conditioning input** — it's a discrete-moment team-shape count,
  not a `pressure_on_actor` parameter. Redirect it: (a) a citable precedent for the **ASI box-defence
  proposal** (⚠ its geometry anchors on the ball-loss location and outfield players, not the GK — GK-anchoring
  needs adapting), and (b) optionally a standalone team-shape KPI beside `compute_pressing_kpis`. It does not
  belong in P1/P1b.
- Any number in an external artifact comes from public data only; apply the full research-discipline doc.
- This maturation is a silly-kicks cycle → its own spec/plan/review, one feature branch, human-gated commit.
  ⚠ Coordinate with the parked TF-58 branch (it edits `_provider_visibility.py` / team-shape / defensive-
  line); do not build on its uncommitted tree.

## 7. Sources

Paper corpus (external, NOT redistributed here — third-party copyright; full citations in §8), from the
PressureBench-TRACE repo: Andrienko 2017 (`dmkd17.pdf`),
Bekkers 2025 (`Pressing_Intensity...pdf`), Merckx 2021 (`merckx-mlsa21-pressing.pdf`), Forcher 2022
(`The keys of pressing...pdf`), Forcher scoping review, Calabuig 2024 (`mathematics-12-03854.pdf`), Player
Pressure Map (`68d6be...pdf`), Dash 2025 (`2511.06191v1.pdf`), Nazarudin 2025 (`Art 116.pdf`), Markou 2024
KTH (`FULLTEXT01.pdf`).

Online (URLs; fetch before load-bearing citation): Bekkers arXiv:2501.04712; Narizuka arXiv:2606.09452;
Bischofberger arXiv:2606.19931; exPress (MIT Sloan 2025); Bauer & Anzer 2021 (DMKD, counterpressing
detector); Robberechts 2019 / Merckx 2021 (VPEP); Gu et al. Player Pressure Map arXiv:2401.16235; Peters et
al. 2025 (IJPAS, paywalled); CMU SURE 2025 "Forced Turnover"; SkillCorner "Pressing Playmakers".

## 8. Citation due-diligence (verified 2026-09-30)

| Item | Verified citation | Access | Link |
|---|---|---|---|
| **Bekkers "Pressing Intensity"** | Bekkers, J. (2025). *Pressing Intensity: An Intuitive Measure for Pressing in Soccer.* arXiv **2501.04712** (v2, 30 Jun 2025), UnravelSports. | FREE · **full text read 2026-09-30** | https://arxiv.org/abs/2501.04712 |
| **Spearman et al. 2017 (pass-prob parent)** | Spearman, W., Basye, A., Dick, G., Hotovy, R., & Pop, P. (2017). *Physics-Based Modeling of Pass Probabilities in Soccer.* 11th MIT Sloan SAC. — the shared parent of both `bekkers_pi` and Spearman-2018 PPCF. ⚠ author order inconsistent across citing refs. | FREE (findable) | MIT Sloan proceedings |
| ⚠ NOTICE typo | `main:NOTICE` cites Bekkers as arXiv **2501.00712** — that id is "Rethinking Addressing in Language Models…" (unrelated). Wrong by one digit. | fix in silly-kicks cycle | (context: https://arxiv.org/abs/2501.00712 = the wrong paper) |
| **Narizuka min-arrival-time** | Narizuka, T., Sakamoto, I., Yamamoto, K., & Yamazaki, Y. (2026). *Quantifying defensive pressure on the ball carrier in soccer based on minimum arrival time.* arXiv **2606.09452** (v2). Fujimura–Sugihara damped-drive motion model (α=1.0 s⁻¹, Vmax=10 m/s fixed from Narizuka 2023); 306 J-League 2023 matches, dual outcome channels (progression at start / loss at release), zone-stratified. | FREE · **full text read 2026-09-30** | https://arxiv.org/abs/2606.09452 |
| **Peters rest-defence** | Peters, A., Parmar, N., Davies, M., & James, N. (2025). *A rule-based approach to classify counterpressing – analysis of its risks and relationship with rest defence.* IJPAS 26(1), 206–222. DOI **10.1080/24748668.2025.2473799**. 380 EPL 2020/21, 12,460 possessions; rest-defence count (central strip ∩ 30 m of loss, final-third losses) predicts shots-conceded/final-third-entries but NOT counterpress initiation. | obtained · **full text read 2026-09-30** | https://www.tandfonline.com/doi/full/10.1080/24748668.2025.2473799 |
| **Spearman pitch control** | Spearman, W. (2018). *Beyond Expected Goals.* 12th MIT Sloan SAC, 1–17. PPCF integral; Bayesian-MAP params (Table 1). ⚠ extracted accel/max-speed look unit-transposed — verify before hard-coding. | FREE · **full text read 2026-09-30** | ResearchGate / MIT Sloan proceedings |
| **Fernández & Bornn** | Fernández, J., & Bornn, L. (2018). *Wide Open Spaces: A statistical technique for measuring space creation in professional soccer.* 12th MIT Sloan SAC. Positional-influence PC surface (expert-elicited, NOT fit); no outcome validation ("no ground truth"). | FREE · **full text read 2026-09-30** | https://www.researchgate.net/publication/324942294 |

Author-attribution note: "Pressing Intensity" is **Bekkers (UnravelSports)**, NOT Bauer & Anzer — Bauer &
Anzer (2021, DMKD) is the *separate* supervised counterpressing-detector paper. Keep them distinct.
