# xT-GK v2 — Sub-project 1: V(z,p), the Honest Possession-Value Surface

**Status:** DRAFT (rev 4 — §M1 reconciled to the implementation's per-action conditioning, D2; specs are not versioned in this repo). NOT committed.
**For:** silly-kicks session (owns implementation).
**From:** Bounou/xT-GK collaboration session (design + cross-repo readiness audit).
**Date:** 2026-07-05
**Governing docs:** `ADR-024` (xt-gk v1), `ADR-021` (value-iteration convergence — monotone-from-below stop condition), `ADR-028` (acting-team-LTR orientation), `ADR-011` (trained-model lifecycle), Eyestone v2 spec `xT-GK_v1-to-v2_Migration_Spec.md` (in `…\project xT-GK\Re GK metrics collaboration - 2026-07-05.eml`).
**Related:** this is **1 of 5** v2 sub-projects. The others (V_opp, ρ/xR-GK, metric+lakehouse migration, validation suite) get their own specs. Do not build them here.

### Revision note (rev 1 → rev 2)
The silly-kicks review found the rev-1 reward mechanism **factually wrong** and it has been redesigned. Summary of what changed, with the verified evidence:
- **Reward is no longer the `vaep.labels` possession-window surface.** That function (`_scores_possession`, `labels.py:264–306`) is **goal-gated** — it credits xG only when a goal is actually scored (`labels.py:282,286`), so it returns a P(goal)-weighted quantity that is near-zero and flat in deep zones — i.e. it *reproduces* the v1 degeneracy the gate exists to catch. Rev 1's "already supported / locked" claim was false.
- **Production reward is now an xG-calibrated *immediate* reward inside the existing Markov solver** (`E[xG|shot]·P(shot)`), a one-surface substitution into tested code. This also removes rev 1's double-count (a terminal reward fed through Bellman propagation) and drops `add_possessions`/the DTAI possession-definition question from the production path entirely.
- **Cross-check is now a model-free empirical possession-xG surface** (not a learned booster — a booster on the same target shares the primary's blind spot; owner decision 2026-07-05).
- **The make-or-break gate is now pre-registered** with numeric acceptance criteria (owner/Eyestone to lock the numbers before fitting).
- Added: attack-orientation fit contract (M4), v1/v2 home end-state (M5), CI-vs-owner-run split (N7), and fixes N1–N6.

---

### Revision note (rev 2 → rev 3)
Review #2 confirmed rev 2 resolved all rev-1 blockers and raised four targeted items, all verified in source and adopted:
- **G1 (functional mismatch, was a gate bug).** The Markov solver computes `E[xG of the possession's *first* shot | z,p]` (the shoot branch carries no continuation; progression requires the move branch — `_value_iteration.py:50–51`, `_action_prob` `p_shot+p_move=1` `_grid.py:154–161`). The cross-check therefore gets a matching **`first_shot`** aggregation as its default; `noisy_or`/`sum` are reported alternatives only. §8.5 reframed: the cross-check validates the gradient on **build-up cells that feed the deep zone**, not absolute agreement at deep cells (where immediate reward ≈0 and V is *pure propagation*).
- **G2 (train/serve gap on the primary use-case).** The reused `_get_move_actions` is `pass ∪ dribble ∪ cross` only (`_grid.py:105–109`) — **goal-kicks are structurally excluded**, yet goal-kick distribution is xT-GK's headline. Decision: **extend the move-set for this surface to include `goalkick` (and `throw_in`)**, denominators updated consistently, with a guard test that goal-kicks appear in `T_p`. (Eyestone confirm — Q5.)
- **G3 (per-cohort vs pooled).** Gate runs **per-cohort** (two independent confirmations); production surface **per-cohort by default**; pooling only after a **cross-provider comparability gate** (pressure + xG scale parity), mirroring the ADR-024 SkillCorner-vs-GS precedent.
- **G4 (phantom persistence).** `ExpectedThreat` has no `save`/`load`; the artifact convention is the pickle-free **npz/JSON + SHA256 + provenance** pattern (ghost-GK/xShot/xCross/`GkCompletionModel`, ADR-011/016).
- Folded in: input-contract validator (G5), tercile-mode-conditional test (G6), negative-control honesty test (G7), deep-zone-specific pressure-coverage check (G8), cross-check partial-port (G9), orthogonal-guards note (G10).

---

## 0. Context — why this exists

Jeffrey Eyestone delivered **xT-GK v2** after our two-cohort empirical finding (WC2022/Gradient + Real-Madrid/SkillCorner) that v1's additive composite is **near-constant across keepers** — a *formulation-bound* degeneracy, not a data artifact. He conceded and rebuilt the metric. v2 discards the v1 additive composite (`Base + PEV + RAV − C_risk`), the `xT·φ` revaluation, all six hand-set params (α,β,γ,δ,λ,ω), and the Option-B origin patch, replacing them with one expectation:

```
xT-GK(s,a) = ρ·[V(s′) − V(s)]  −  (1−ρ)·[V(s) + κ·V_opp(s,a)]
```

- `ρ(s,a)` — success probability from a calibrated **retention classifier** (v2: *do NOT use raw completion*). Decision: we build our own calibrated `P(retain|s,a)` behind a swappable interface (Eyestone's xR-GK drops in later). Separate sub-project.
- `V(·)` — **the honest possession-value surface. THIS SPEC.**
- `V_opp(s,a)` — turnover cost (v1's 180°-mirror `_counter_value` in `_xt_gk.py` is the proxy it replaces). Separate sub-project.
- `κ ≥ 1` — the only free scalar (default 1; report sensitivity across [1,2]).

**Why V(z,p) is sub-project 1 and gates everything:** v1 degenerates because its value surface is the raw destination-only Singh xT, ≈0 and flat in the keeper zone (documented in `_xt_gk.py`; `CLAUDE.md`: deep raw xT ≈ 0.0085). Eyestone §2.1 names V "the highest-leverage change" with a **make-or-break diagnostic**: if the deep zone shows no real, pressure-dependent gradient after V is refit honestly, *v2 cannot separate keepers* and we stop. We build V first, cheaply prove or kill the gradient on real data, and only then fund sub-projects 2–5.

---

## 1. Purpose & scope

**Purpose.** Produce `V(z, p)` = the **expected xG the possessing team generates over the remainder of the current possession**, given the ball in 16×12 grid-zone `z` under pressure level `p ∈ {1,2,3}` (low/med/high). Replaces v1's destination-only raw-xT surface as the value function inside xT-GK.

**In scope (this spec):** the production `V(z,p)` estimator; a model-free empirical cross-check; the `PossessionValue` port + adapters; the `delta_v` decomposition hooks; the pre-registered deep-zone gate; the test strategy.

**Out of scope (other sub-projects):** the v2 metric assembly / `κ` / PEV·DZV·RAV *reporting*; `ρ`; `V_opp`; the lakehouse migration (`enrich.py`, `fct_action_context` columns, dbt, UI); reconciling the lakehouse's separate 12×8 general-play xT (`gk_xt_delta`) with the 16×12 grid (tech-debt, untouched here).

---

## 2. Locked design decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Production estimator** | Pressure-stratified **Markov value iteration with an xG-calibrated immediate reward** | Reuses the tested solver almost byte-for-byte; classic xT already IS "expected possession outcome via undiscounted Markov propagation with turnover absorption" (see §4). Decomposable at arbitrary `(z,p)` for §7 Shapley. Spec-faithful ("Markov value iteration in Singh's spirit"). |
| **Reward** | **`E[xG|shot]·P(shot)`** per `(z,p)` cell (immediate expected xG), propagated by the solver | Turns Singh's `P(goal)` into expected-xG-over-remainder-of-possession by one surface substitution (§4.1). NOT the goal-gated `vaep.labels` possession surface. |
| **Cross-check** | **Model-free empirical** `E[possession-xG \| z,p]`, **`first_shot` aggregation** (default) | Must estimate the SAME functional as the Markov surface (first-shot xG) or §8.5's agreement test is ill-posed (G1). Independent estimator (unbiased at `(z,p)`, high-variance) vs Markov (regularized, Markov-bias); disagreement on the build-up gradient catches transition-structure artifacts. `noisy_or`/`sum` reported as alternatives only. |
| **GK action population** | move-set = `pass ∪ dribble ∪ cross ∪ goalkick ∪ throw_in` | Classic xT excludes restarts; this is a *new* GK surface and goal-kick distribution is the headline use-case, so goal-kicks MUST be in the transition law (G2). Denominators updated consistently; guard test asserts goal-kicks in `T_p`. Eyestone-confirm (Q5). |
| **Surface scope** | **Per-cohort** (WC2022, RM); pooling only behind a comparability gate | Gate wants two independent confirmations; provider non-comparability is documented here (ADR-024). Pooled terciles/xG assume cross-provider scale parity — not free (G3). |
| **xG input** | **Injected `xg_column`; silly-kicks ships NO xG model** | silly-kicks deliberately has no pre-shot xG (`_xshot_occurrence` = P(shot), xG "out of scope"; `_shot_goalmouth` = post-shot PSxG, leakage-flagged). Mirrors how v1 injects a pre-fit `ExpectedThreat`. Source deferred to owner (§6, Q3). |
| **Grid** | **16×12 (Singh)** | Eyestone §2.1 + `xthreat` default (`GridSpec`). |
| **Pressure levels** | `{1,2,3}` via terciles of one continuous measure; **occupancy reported** (§5, M3) | Eyestone §2.1; data-driven cutpoints. |
| **Horizon / discount** | Possession-scoped, undiscounted (matches `value_iteration`, ADR-021) | The turnover-absorbing transition matrix bounds the horizon implicitly (§4). |
| **Home** | **silly-kicks** (`silly_kicks/xtgk/…`) | Source of truth for metric surfaces; lakehouse only materializes. End-state in §M5. |

---

## 3. Architecture

Hexagonal. One port, two adapters — the metric layer depends on the interface, not an estimator.

```
                       ┌────────────────────────────────────────┐
                       │  PossessionValue (Protocol)             │
                       │    value(zone, p) -> float              │
                       │    surface(p)     -> ndarray (W, L)     │
                       │    delta_v(s, s') -> DeltaV (Shapley)   │
                       └───────────────┬───────────────┬─────────┘
                                       │               │
             ┌─────────────────────────▼──┐   ┌────────▼──────────────────────────┐
             │ MarkovPossessionValue       │   │ EmpiricalPossessionValue           │
             │  (PRODUCTION)               │   │  (CROSS-CHECK ONLY, not shipped)   │
             │  xG-calibrated immediate    │   │  model-free E[possession-xG|z,p]   │
             │  reward + pressure-strat.   │   │  over ALL shots (sum / noisy-or)   │
             │  value iteration            │   │  via add_possessions               │
             └─────────────────────────────┘   └────────────────────────────────────┘
```

**Proposed module layout** — new `silly_kicks/xtgk/` package (v2 gets its own namespace; do **not** overload `tracking/_xt_gk.py`, which is v1 and still consumed):

```
silly_kicks/xtgk/
  __init__.py
  _possession_value.py     # PossessionValue Protocol + shared types (PressureLevel, State, DeltaV)
  _markov.py               # MarkovPossessionValue: fit / value / surface / delta_v
  _xg_reward.py            # per-cell E[xG|shot] surface from an injected xg_column (mirror _grid._scoring_prob)
  _pressure_levels.py      # continuous pressure -> {1,2,3} tercile quantizer (fit cutpoints, apply, occupancy report)
  _empirical.py            # EmpiricalPossessionValue cross-check (per-action first-shot xG; O(n) reverse scan)
  _diagnostics.py          # pre-registered deep-zone gate + per-cell support / tercile-occupancy reports
```

Note (N6): pressure stratification is *not* a new transitions module — it is `singh_transition_matrix` / `_action_prob` / the xG-reward surface each called on the pressure-`p` **subset** of actions. A thin helper, not `_transitions_pressure.py`, until it earns a file.

**Dependency direction (M6):** the **production** estimator depends only on `xthreat` + `spadl` + the injected `xg_column` — *no* `vaep.labels` coupling. The only `vaep`/`add_possessions` use is inside the **cross-check** (`_empirical.py`), a deliberate, documented import, not a structural dependency of production.

**Reuse map (extend, don't fork):**

| Need | Reuse | Note |
|---|---|---|
| Markov solve | `xthreat/_value_iteration.py::value_iteration` | Verbatim. `gs = p_scoring·p_shot` becomes `E[xG|shot]·P(shot)`. Called once per pressure level. ADR-021 stop condition preserved (N3). |
| Immediate reward | pattern of `xthreat/_grid.py::_scoring_prob` (`:80–85`) | New `_xg_reward`: replace `goalmatrix` (goal counts) with a per-cell **xG sum over shots**; divide by shot counts ⇒ `E[xG|shot]`. |
| Shot/move split | `xthreat/_grid.py::_action_prob` (`:131–161`) | On the pressure-`p` subset. |
| Transition + turnover absorber | `xthreat/_transitions.py::singh_transition_matrix` (`:12–46`) | Rows sum <1 by construction (denominator = all moves, numerator = successful; `:36–45`) — the missing mass is the turnover-to-zero-value absorber. On the pressure-`p` subset; KDE path (`kde_smoothed_transition_matrix`) for sparse cells. |
| **Move-set (G2)** | **an *extended* `_get_move_actions`** | Stock `_get_move_actions` (`_grid.py:105–109`) = `pass ∪ dribble ∪ cross`, which **excludes goal-kicks/throw-ins**. This surface needs `∪ goalkick ∪ throw_in`. Because `singh_transition_matrix` and `_action_prob` both call `_get_move_actions` internally, the clean seam is a new move-set function (or an injectable predicate) used by the `xtgk` reward/transition builders — do NOT edit `xthreat`'s `_get_move_actions` (regression boundary). |
| Grid / cell indexing | `xthreat/_params.py::GridSpec`, `_grid.py::_get_cell_indexes` | 16×12; shared indexing keeps orientation conventions aligned. |
| Cross-check aggregation | `spadl/utils.py::add_possessions` + a **first-shot** xG aggregation | Match the Markov functional (G1): **per action**, the xG of the **first shot after it** within the possession (0 if none), averaged per `(cell, tercile)` — conditioned per-action, not per-possession-origin (§M1/D2). The stock `_scores_possession` goal-gating must NOT be reused; `noisy_or`/`sum` are reported alternatives only. |

---

## 4. Production estimator — `MarkovPossessionValue`

Classic xT is already an undiscounted Markov possession-value model: `V = gs + p_move·(T·V)`, with loss-of-possession absorbed by the sub-stochastic `T` (§3 reuse table). Three changes, nothing else:

**4.1 Reward = expected xG, not P(goal).** Build a per-cell immediate reward `xg_scoring(z,p) = E[xG | shot in cell z, pressure p]` from the injected `xg_column`:
`xg_scoring = (Σ xg of shots in cell) / (shots in cell)` — the one-surface analogue of `_scoring_prob`'s `goals/shots`. Then `gs(z,p) = xg_scoring(z,p) · p_shot(z,p)` = immediate expected xG. This is the ONLY reward; there is no separate terminal-value aggregation (rev-1's double-count is gone). Deep build-up inherits realized xG *through the Markov propagation*, which is the honest surface.
The `p_shot`/`p_move` denominators and the transition matrix are all computed over the **extended move-set** (incl. goal-kicks/throw-ins, §G2/§3) so the reward, action-choice split, and transition law describe the same action population — the one the metric actually scores.

**4.2 Pressure stratification → three surfaces.** Partition actions by `p ∈ {1,2,3}`; build `xg_scoring_p`, `p_shot_p`, `p_move_p`, `T_p` on each subset; run `value_iteration` per level → `V(·,1), V(·,2), V(·,3)`.

**What V actually estimates (G1 — carry this into the gate).** Because the shoot branch is terminal (no continuation in `value_iteration`) and progression requires the move branch, the recursion values the **first shot the possession reaches**:
`V(z,p) = Σ_c P(reach c from z under p without shooting earlier) · p_shot(c) · E[xG|shot at c]` = `E[ xG of the possession's first shot | z, p ]`. Two consequences the gate must respect: (i) the cross-check must estimate the *same* functional (`first_shot`, §M1), not all-shots; (ii) **deep-zone V is pure propagation** — deep cells have ≈0 shots, so `E[xG|shot at deep cell] = 0/0 → 0` (`_safe_divide`, `_grid.py:59–60`) and their value comes entirely from `p_move·T·V` carrying forward value. So the gate is really testing *"do pressure-stratified transitions give deep possessions different forward shot-access?"* — `_diagnostics.py` reports the per-tercile `T_p` deep→forward mass so the mechanism is visible, not just the output number.

**Per-level vs joint `(z,p)` (M2 — a correctness choice, not a knob).** Solving each `V(·,p)` with pressure held fixed for the whole possession assumes pressure does not change within a possession — it always does. **Default: per-level surfaces, with the approximation stated openly** — each `V(·,p)` is "possession value *assuming pressure stays at p*"; cross-pressure effects enter only at `delta_v` via `(z,p)→(z′,p′)` corner evaluation. Escalate to a joint `(z,p)` state space (correct, but sparser — aggravates the gate's support needs) **only if** the gate is marginal under per-level. Pre-register (open item #1) which the fit data actually supports before fitting. This is an Eyestone question (Q2).

**4.3 Sparsity, fail-loud.** Deep + high-press cells are sparse. Use the existing `kde_smoothed` transition builder; `_diagnostics.py` **reports per-cell support**. For general cells, warn-and-count below a support threshold (house fail-safe style). For **gate cells specifically, insufficient support is a STOP, not a warn** (§8) — the gate cannot run on empty cells.

**Fitted artifact (G4 — real anchor).** `ExpectedThreat` has **no** `save`/`load`, so there is no "xT-grid convention" to reuse. Persist with the house **pickle-free fitted-artifact** pattern (ghost-GK / xShot / xCross / `GkCompletionModel`; ADR-011/016): `npz`/JSON payload + `SHA256SUMS` + provenance `metadata.json`. Contents per cohort: the three surfaces, the pressure cutpoints, the `xg_column` identity, per-cell support counts, and cohort/grid provenance. This is a *fitted grid*, not an ADR-011 weights lifecycle — but the same serialization convention (no pickle) applies.

---

## 5. Pressure discretization — `_pressure_levels.py`

- Input: one continuous pressure measure per action. **Default: the measure v1 consumes** (`pressure_on_actor(...)` in `_xt_gk.py`); the choice among `andrienko_oval` / `link_zones` / `bekkers_pi` is pinned at build time by *which yields the clearest deep-zone gradient* (§8). Record the winner + rationale.
- Quantize to `{1,2,3}` by terciles; `fit()` learns cutpoints, `apply()` maps new actions; **persist cutpoints with the surface** (a keeper scored on WC cutpoints must not be silently re-terciled on RM).
- **Occupancy reporting (M3, required before trusting the gate):** pressure is zone-dependent — deep zones are plausibly systematically low-pressure, so *global* terciles may push nearly all deep-zone actions into one tercile, leaving `V(deep, p=3)` estimated on almost nothing and the gate measured across an empty stratification. `_pressure_levels` must emit the **within-deep-zone tercile occupancy** (counts per `(deep-cell, tercile)`). If deep-zone occupancy is degenerate, use **zone-conditional terciles** (terciles computed within zone bands) and record that choice. This interacts with the §8 support floor.
- **Missing-pressure handling (refined by G8, 2026-07-09 — NOT a blanket drop).** Distinguish two cases by tracking-frame presence: **(a) frame absent** (genuine tracking gap) → fail-loud / drop, never default a level; **(b) frame present but `pressure_on_actor` null** → the actor is *genuinely unpressured* (no opponent inside the pressure region) = **zero pressure → LOW tercile, keep it.** G8 verified case (b) is exactly the GS goal-kick situation: **595/595 GS null-pressure goal-kicks have intact frames (~10 defenders detected)**. A blanket fail-loud-drop would silently discard 60% of WC goal-kicks — the certified cohort's headline population. Report the null-pressure-with-frame count per cohort (it is signal, not loss). This also empirically confirms the M3 note that deep zones skew low-pressure.

**Attack-orientation fit contract (M4 — the repeatedly-burned failure mode).** V is only meaningful if fit on **attack-oriented** actions (acting team attacks x=105, so "deep" = own goal). `fit()` **requires attack-positive actions** and must guard it: assert the cohort feed is acting-team-LTR (ADR-028; both GS and SkillCorner feeds are already reprojected — confirm per cohort) and fail loud on a mixed/half-mirrored corpus. A silently half-mirrored fit inverts deep/final-third and would fake or destroy the gradient. (Contrast: the §8 *gate* deliberately runs in both orientations to test robustness; the *fit input* must be single, correct orientation.)

---

## 6. Data, cohorts, prerequisites

- **Fit cohorts, per-cohort surfaces (G3):** WC2022 (`data_source=gradientsports`) and RM (`data_source=skillcorner`, `access_tier=restricted`) each get their **own** surface; the gate confirms the gradient on each independently. Verified live: 89,909 and 122,426 action-context rows; pressure populated in aggregate. Fit each surface on **full possession chains including all outfield build-up AND keeper distributions (goal-kicks/throw-ins, per §G2)** so deep zones are populated by the actions the metric actually scores.
- **Pooling gate (G3):** a pooled cross-cohort surface is *not* produced by default. If one is wanted later, it is gated by a **cross-provider comparability check** — pressure-distribution parity and injected-xG scale parity across GS and SkillCorner — mirroring the ADR-024 SkillCorner-vs-GS `xt_gk` comparability precedent (separate completion variants were needed there). Absent that check, terciles and reward are silently miscalibrated across cohorts.
- **Deep-zone pressure coverage (G8 — RESOLVED 2026-07-09, live-verified):** open-play deep-zone actions are **100% pressure-covered in both cohorts** (GS 6,084/6,084; SC 12,834/12,834) — the gate's build-up cells are safe. Goal-kicks: SC 100% (1,181/1,181); **GS 393/988 carry a positive pressure value, the other 595 are null — but all 595 have intact tracking frames (~10 defenders) → genuinely *unpressured*, not missing.** Per the §5 frame-aware rule they map to the LOW tercile (do NOT drop; a blanket drop would have lost 60% of WC goal-kicks). All three pressure measures share identical coverage (coalescing recovers nothing). **Net: no coverage blocker** — the only actions are the §5 frame-aware null-pressure handling and reporting the unpressured-goal-kick count per cohort.
- **Metrica excluded** (GK actions unscored in current gold; stale-data deferral).
- **Reward input dependency (the one true external need) — B2/Q3, RESOLVED 2026-07-09:** the injected `xg_column` is **`soccer_analytics.dev_gold.fct_shot_xg.xg`** — documented "Calibrated pre-shot xG mean (0-1)", grain `(match_key, action_id)` (joins the fit input directly; a *separate* table from `fct_shot_psxg`, so no post-shot leakage), with `xg_ci_low/high` CIs and per-shot quality flags (`scoring_mode`, `ood_flag`). silly-kicks still ships no xG model — this is injected. **Live coverage (verified):** 100% non-null on both cohorts (WC2022/GS 1473/1473 shots·64 matches; RM/SC 2596/2596·108 matches). **⚠ Certification caveat (only the backend check surfaced this):** `ood_flag` = **0 for gradientsports (fully certified)** but **100% for skillcorner (all 2596 RM shots out-of-distribution / uncertified)** — the xG model is OOD on SkillCorner's tracking-derived shot features. Because surfaces are per-cohort (G3), this is contained: the **WC reward is certified (clean go/no-go)**, the **RM reward is populated-but-uncertified (RM verdict provisional)**. Required of the fit/owner-run: **carry `ood_flag` + CIs into provenance**, **report per-cohort OOD-rate alongside the §G8 pressure-coverage report**, and **caveat RM gate results** (weaker evidence; candidate for an xg-CI sensitivity pass). Feeds the Jeff/Q4 discussion — "how much RM separation is real" is softened by an uncertified RM reward. (Minor QC: GS `min_xg = 0.0` — a few exact-zero-xG shots; eyeball, but they only contribute 0 reward.)
- **No `add_possessions` dependency in production** (removed with the reward redesign). `add_possessions` is used only by the cross-check (§M1); if that path needs a possession definition, DTAI-naive vs DTAI-extended is a **cross-check-local** choice, recorded there — it no longer gates production.

---

## 7. Decomposition hooks — `delta_v`

The v2 metric's §4 keeps the four-bar chart as *honest decompositions* via a two-factor Shapley split on `ΔV = V(z',p') − V(z,p)`:

```
ΔV_pressure = ½[(V(z,p') − V(z,p)) + (V(z',p') − V(z',p))]
ΔV_position = ½[(V(z',p) − V(z,p)) + (V(z',p') − V(z,p'))]
ΔV_position + ΔV_pressure = ΔV
```

`delta_v(s, s')` exposes the four corners `V(z,p), V(z,p'), V(z',p), V(z',p')` and returns the split. Build it into the port now so the metric layer never reaches into estimator internals.
**Test the right thing (N2):** the identity above is near-tautological (the symmetric two-factor Shapley telescopes to `ΔV` by construction) — keep it as a cheap regression guard, but the *real* risk is a corner hitting an **unsupported** `(z,p')` cell. Test that path: graceful handling or STOP, per §4.3/§8, not just the algebra.

---

## 8. The make-or-break gate — `_diagnostics.py` (BLOCKING, PRE-REGISTERED)

The single most important test here; it gates four downstream sub-projects, so it must be **pre-registered before fitting** — no post-hoc "monotone-ish". Owner/Eyestone lock the numbers (Q4); the *structure* is fixed here:

1. **Deep-zone cell set** — *proposed default:* the 2 grid columns nearest the acting team's own goal (attack-LTR x-cells 0–1 of 16), all 12 rows. Lock the exact set before fitting.
2. **Effect-size floor** — `|V(deep, p=hi) − V(deep, p=lo)|` must exceed an absolute floor **and** ≥ 2× the pooled per-cell standard error (so a "gradient" built on noise fails). *Proposed:* floor tied to the magnitude the metric must explain across keepers; Eyestone/owner set the number (Q4). A non-zero-but-trivial gradient is a FAIL.
3. **Monotonicity** — pressure-ordered within tolerance. *Expected direction:* value **decreases** as pressure rises (pressure suppresses progression/retention), i.e. `V(deep, p=1) ≥ V(deep, p=2) ≥ V(deep, p=3)` — **confirm the direction with Eyestone** (Q2); a real gradient of the wrong sign is itself a finding to escalate, not silently pass.
4. **Gate-cell support minimum** — each gate cell (and each tercile within it, per §5/M3) needs ≥ N_min observations; **insufficient support in a gate cell ⇒ STOP** (the gate cannot run), not warn-and-proceed. Lock N_min before fitting.
5. **Cross-check agreement (G1 — on build-up cells, not deep cells)** — compare the Markov surface to the `first_shot` empirical surface on the **build-up cells that feed the deep zone** (the mid cells whose forward transitions carry value into deep possessions), agreeing on sign and relative shape within a stated tolerance. Do **not** grade agreement at the deep cells themselves: there both surfaces are ≈0 (pure propagation / no immediate shots), so "agreement at ≈0" is trivial and uninformative. Divergence on the build-up gradient is a red flag to investigate, not a pass.
6. **Run conditions (G3 — per-cohort)** — run the gate **independently on WC2022 and on RM** (two independent confirmations are far stronger than one pooled result), each in **both pitch orientations** (robustness). Production-realistic fixtures only; emit the support, tercile-occupancy, and `T_p` deep→forward-mass reports alongside each verdict. A pooled surface is NOT gated here (see §6/G3).

**Outcome:** PASS ⇒ authorise sub-projects 2–5. FAIL (flat / trivial / unsupported / cross-check divergent) ⇒ **STOP and escalate to owner + Eyestone**; do not build V_opp/ρ. Failing here costs one sub-project instead of five — that is the point.

---

## 9. Interfaces (indicative — silly-kicks session finalises)

```python
PressureLevel = Literal[1, 2, 3]

class PossessionValue(Protocol):
    def value(self, zone: int, p: PressureLevel) -> float: ...
    def surface(self, p: PressureLevel) -> npt.NDArray[np.float64]: ...   # (W, L)
    def delta_v(self, s: State, s_next: State) -> DeltaV: ...             # ΔV + Shapley split

class MarkovPossessionValue:               # implements the full port
    def fit(self, actions: pd.DataFrame, *, xg_column: str,
            pressure_column: str, eps: float = 1e-5) -> "MarkovPossessionValue": ...
    # value/surface/delta_v per the port; move-set incl. goalkick/throw_in (§G2)

class EmpiricalPossessionValue:            # cross-check only, not shipped; surface/value ONLY (no delta_v, G9)
    def fit(self, actions: pd.DataFrame, *, xg_column: str, pressure_column: str,
            aggregation: Literal["first_shot", "noisy_or", "sum"] = "first_shot") -> "EmpiricalPossessionValue": ...
```

`State` is a small dataclass carrying at least `(zone, pressure_level)` plus what `delta_v` needs to locate the four corners. `fit`-before-use raises (see N1).

---

## M1. Cross-check — `EmpiricalPossessionValue` (model-free)

Condition on `(cell, tercile)` **per action (D2, accepted 2026-07-05)** — every action's own cell+tercile, with outcome = the xG of the first shot *after that action* within the possession (0 if none). Rationale: the Markov surface estimates the value of the ball being at `z` **now**, not of a possession that *originated* at `z`; so the like-for-like empirical estimator must condition on **every action whose ball is at `z`**, not just possession-start actions. (Per-origin conditioning — the earlier wording — would also **starve deep cells**: a deep cell would be credited only for possessions that *began* there, not for every possession whose ball passed through it.) Requirements:
- **`first_shot` aggregation is the DEFAULT (G1)** — the like-for-like estimator of the Markov target: **per action**, the xG of the **first shot after it** within the possession (0 if none); averaged per `(cell, tercile)`. This is the only aggregation used in the §8.5 comparison. `noisy_or` = `1 − Π(1 − xg_i)` and `sum` (over all shots) are **reported alternatives** for context, never the comparison target (they estimate a different functional and would manufacture/mask §8.5 disagreement).
- **Never goal-gated.** Do **not** reuse `_scores_possession` (it maxes xG *only over goals*, `labels.py:282,286`).
- **Genuinely independent of the Markov estimator** — no shared transition structure, no shared reward-propagation. That independence is what makes disagreement diagnostic.
- **Partial port (G9).** It implements `surface`/`value` only; it is a throwaway validator and does **not** implement `delta_v`. The `PossessionValue` Protocol is intentionally partially-implemented for this adapter (mark it in code).

**Orthogonal guards (G10).** Both estimators consume the *same* injected `xg_column`, so §8.5 validates **transition/propagation structure**, NOT xG calibration — calibration is the §6 pre-fit gate's separate job. "Cross-check passed" does **not** imply "reward is calibrated"; the two guards are orthogonal and both required.

Not shipped, no weights, no ADR-011 lifecycle — it validates the gate and is retired after.

---

## M5. v1 / v2 home — intended end-state

To avoid permanent split-brain between `tracking/_xt_gk.py` (v1) and `silly_kicks/xtgk/` (v2):
- **v2** ultimately hosts the full metric under `silly_kicks/xtgk/` (V here; V_opp, ρ-interface, and the metric assembly in later sub-projects).
- **v1** `tracking/_xt_gk.py` is **frozen** on landing of the v2 metric (sub-project 4) and **deprecated** once the lakehouse migrates its `xt_gk*` columns to v2 output; removed no earlier than one release after the lakehouse cutover (Hyrum's-Law grace — the GK-Analytics UI reads v1 columns today).
- Until then, both coexist; `CLAUDE.md` documents the transition. No v2 code imports v1 and vice-versa (the regression gate in §10 enforces `xthreat` byte-stability, not v1 stability — v1 is untouched here regardless).

---

## 10. Testing strategy (TDD / hexagonal / E2E)

Write tests first. CI-vs-owner split is explicit (N7):

**CI (committed, synthetic + units — necessary, not sufficient):**
- Unit — `_xg_reward`: per-cell `E[xG|shot]` = Σxg/shots; NaN-coord shots excluded and counted (mirror `_scoring_prob`/`rate` dropna); **own goals excluded from the reward** (N4 — own goals are not the possessing team's shot xG; they are V_opp's concern; do not import `_is_owngoal` into the reward path).
- Unit — `_pressure_levels`: tercile partition test is **conditional on the active mode (G6)** — global terciles ⇒ ~⅓ each; *zone-conditional* terciles (§5/M3) ⇒ ~⅓ *within each zone band*, NOT globally (a global-⅓ assertion would falsely fail or falsely lock global mode). `apply()` stable to persisted cutpoints; occupancy report emitted; missing pressure raises.
- Unit — stratified transitions: `T_p` rows sub-stochastic (turnover mass present); KDE path invoked for sparse cells; support counted; **guard: goal-kicks/throw-ins appear in `T_p` (G2)** — assert non-zero goal-kick rows, i.e. the extended move-set is actually in the transition law.
- Unit — input validator (`validate_possession_value_input`, G5): raises loudly on missing `xg_column`/`pressure_column`, wrong dtypes, non-attack-LTR orientation (§M4), and (cross-check) missing `possession_id`.
- Unit — solver reuse: per-level `value_iteration` converges; matches a hand-computed small grid (golden); **note in-code that the monotone-from-below stop condition holds under an xG≥0 reward — do NOT "fix" it** (N3, ADR-021).
- Property — **honesty + negative control (both orientations, G7):** a synthetic cohort where deep possessions demonstrably reach shots ⇒ deep `V` > 0 and rises as build-up value rises; **and the paired negative control** — a synthetic cohort where deep possessions do NOT reach shots ⇒ deep `V` ≈ 0 and flat. Both are required: without the negative control the test cannot show it *discriminates* (pin-the-ground-truth lesson). CI stand-in for the gate — necessary, not sufficient.
- `delta_v`: Shapley identity regression guard; **unsupported-corner** handling test (N2).
- Fitted-state: `value()/surface()` before `fit()` raises via an explicit `_fitted` flag — **not** `if not np.any(self.xT)` (N1: a legitimately all-zero pressure surface, e.g. a tercile with no shots, would falsely read "unfitted").
- Regression boundary: importing/using `xtgk` leaves `xthreat` outputs byte-identical (classic xT is still consumed elsewhere).

**Owner-run (real data, NOT CI — cohorts are owner/restricted, uncommitted):**
- `@pytest.mark.e2e` + `scripts/validate_xtgk_possession_value.py` (mirror the existing owner-gated `validate_*` pattern) fits on WC2022 + RM, runs the §8 **pre-registered gate**, and emits a report artifact (gradient, effect size, monotonicity, per-cell + tercile-occupancy support, cross-check agreement). This is the real go/no-go; CI cannot run it.

---

## 11. Error handling & fail-safe posture

- `value()/surface()` before `fit()` → `NotFittedError` via explicit `_fitted` flag (N1).
- NaN action coordinates → excluded from cell assignment, counted, never silently zero (mirror `rate`/`_scoring_prob`).
- Unsupported `(z,p)` cell → general cells: warn-and-count with observable signal, never impute silently; **gate cells: STOP** (§8.4).
- Missing pressure / missing `xg_column` → raise with an actionable message (name the column; point at the xG source) — ergonomics of `labels._require_column`.
- Mixed/half-mirrored orientation in the fit corpus → fail loud (§5/M4).
- **Collect these preconditions in one opt-in loud-guard validator (G5):** `validate_possession_value_input(actions, *, xg_column, pressure_column, require_possession_id=False)`, in the house style of `validate_time_base` (ADR-017) / `validate_id_dtypes` (ADR-019) — a single diagnosis object rather than scattered asserts, and the natural home for the §M4 orientation guard.

---

## 12. Open items for the silly-kicks session

1. **Per-level vs joint `(z,p)`** (§4.2/M2) — default per-level with the approximation stated; confirm what the fit data supports; escalate only if the gate is marginal. (Eyestone Q2.)
2. **Canonical `xg_column` source** (§6/B2) — **RESOLVED (Q3, 2026-07-09):** `dev_gold.fct_shot_xg.xg` (calibrated pre-shot, `(match_key,action_id)`); 100% coverage both cohorts; RM shots all `ood_flag`=true (uncertified) → carry flag/CIs, report OOD rate, caveat RM. See §6.
3. **Canonical pressure measure** (§5) — pinned by the gradient check; default v1's.
4. **Cross-check aggregation** (§M1) — `noisy_or` default vs `sum`; record it.
5. **Gate numbers** (§8) — deep-cell set, effect-size floor, monotonicity direction/tolerance, `N_min`, cross-check tolerance — **locked with owner/Eyestone before fitting** (Q4).
6. **Surface artifact format (G4)** — the pickle-free ghost-GK/xShot/xCross/`GkCompletionModel` convention (npz/JSON + SHA256 + provenance `metadata`), NOT a non-existent "xT-grid convention" (`ExpectedThreat` has no `save`/`load`).
7. **GK action population (G2/Q5)** — extended move-set incl. goal-kicks/throw-ins is the default; confirm with Eyestone that valuing keeper distributions this way is faithful to v2 intent.

---

## 13. Definition of done (this sub-project)

- `MarkovPossessionValue` fits three pressure surfaces on WC2022 + RM with reported per-cell + tercile-occupancy support.
- The **pre-registered** deep-zone gate **passes on real data, both orientations**, with the model-free empirical cross-check agreeing within tolerance — OR a documented STOP escalation.
- `delta_v` exposes the Shapley split; identity guard + unsupported-corner test green.
- `xthreat` outputs byte-unchanged (regression gate green).
- CI suite (synthetic honesty + units, both orientations) green; owner-run validation artifact produced.

Nothing ships to the lakehouse or the metric until sub-projects 2–4 land; **V(z,p) proven honest is the gate that authorises that spend.**

---

## Questions to Eyestone / owner (carry into the collaboration call)

- **Q1 (Eyestone).** Confirm "expected possession xG" (v2 §2.1) means the **Markov-propagated immediate-xG surface** (our production estimator), with a **possession-terminal empirical** surface as the model-free cross-check — two estimators of the same quantity. Faithful to intent?
- **Q2 (Eyestone).** Is pressure a **within-possession-varying** state (favouring a joint `(z,p)` Markov chain) or slowly-varying (per-level surfaces acceptable)? And confirm the expected sign of the deep-zone pressure gradient (§8.3).
- **Q3 (owner).** Canonical per-shot xG source — gold `fct_action_context` column vs HF `luxury-lakehouse/xg-model` — and confirm depending on a lakehouse-trained xG *via injection* (silly-kicks ships nothing) respects the "Home = silly-kicks" boundary. (Recommendation: yes.)
- **Q4 (owner/Eyestone).** Lock the §8 gate numbers (deep-cell set, effect-size floor, monotonicity, `N_min`, cross-check tolerance) that you'd accept as "keepers are separable," before we fit.
- **Q5 (Eyestone).** V's transition law will **include goal-kicks/throw-ins** (§G2) so a keeper's goal-kick is valued against a surface that actually contains goal-kicks. Confirm this is the intended representation of keeper distribution in V (vs. an outfield-open-play-only surface with a documented representation gap).

---

## References

- v1 compute: `silly_kicks/tracking/_xt_gk.py` (`compute_xt_gk`, `XtGkParams`, `_counter_value` 180°-mirror = the V_opp proxy).
- xT machinery: `silly_kicks/xthreat/{_model,_value_iteration,_transitions,_params,_grid}.py` (reward pattern `_grid._scoring_prob:80–85`; move-set `_grid._get_move_actions:105–109` = pass∪dribble∪cross, the G2 exclusion; action split `_action_prob:154–161`; turnover absorber `_transitions.py:36–45`; solver `_value_iteration.py:50–51`; `_safe_divide` 0/0→0 `_grid.py:59–60`). NB `_model.py` (`ExpectedThreat`) exposes only `__init__/fit/interpolator/rate` — **no `save`/`load`** (G4).
- Artifact convention (G4) + input validators (G5): ghost-GK/xShot/xCross/`GkCompletionModel` pickle-free npz-JSON+SHA256 (ADR-011/016); `validate_time_base` (ADR-017), `validate_id_dtypes` (ADR-019).
- No pre-shot xG in silly-kicks (verified): `tracking/_xshot_occurrence.py:1–9` (P(shot), xG out of scope), `tracking/_shot_goalmouth.py:1–10` (post-shot PSxG, leakage-flagged).
- Cross-check aggregation base (to be corrected, all-shots): `silly_kicks/vaep/labels.py` (`_scores_possession:264–306` — goal-gated, do NOT reuse as-is), `spadl/utils.py::add_possessions`.
- Pressure: `silly_kicks/tracking/{features.py::pressure_on_actor, pressure.py}`.
- Live gold (later materialisation target): `soccer_analytics.dev_gold.fct_action_context`; Databricks warehouse `soccer-analytics-warehouse-dev`.
- Eyestone v2 spec + full thread: `…\project xT-GK\Re GK metrics collaboration - 2026-07-05.eml`.
