# ADR-036: xT-GK v2 possession-value surface `V(z,p)` (`silly_kicks/xtgk/`)

| Field | Value |
|---|---|
| **Date** | 2026-07-09 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen; Eyestone (xT-GK v2 collaboration); silly-kicks + analysis sessions |

## Context

xT-GK v1 (`tracking/_xt_gk.py`, ADR-024) is **near-constant across keepers** — a formulation-bound degeneracy confirmed on two cohorts (WC2022/GS + Real-Madrid/SkillCorner). Its value surface is the raw destination-only Singh xT, which is ≈0.0085 and flat in the keeper zone, so `V(s') − V(s)` carries almost no signal. Eyestone delivered xT-GK v2, which replaces the v1 additive composite with a single expectation `ρ·[V(s′)−V(s)] − (1−ρ)·[V(s)+κ·V_opp]` whose highest-leverage change is a new value function `V(z,p)` — the expected xG the possessing team generates over the remainder of the possession, given the ball in 16×12 zone `z` under pressure level `p∈{1,2,3}`. This is v2 **sub-project 1 of 5**; it gates the other four via a make-or-break deep-zone-gradient test on real data.

`V(z,p)`'s reward is expected xG, so it needs a calibrated **per-shot xG on every shot** — which silly-kicks deliberately does not ship (it values threat via VAEP/xthreat). xG therefore enters as an **injected `xg_column`**, sourced from the lakehouse `fct_shot_xg` mart. Both fit cohorts are provider-tier/restricted, so the real-data gate is owner-run, not CI.

## Decision

Ship a new hexagonal `silly_kicks/xtgk/` package: a `PossessionValue` Protocol with two adapters — `MarkovPossessionValue` (production: pressure-stratified value iteration reusing `xthreat`'s tested solver, with an xG-calibrated *first-shot* immediate reward and a goal-kick-inclusive move-set) and `EmpiricalPossessionValue` (a model-free `first_shot` cross-check, not shipped) — plus a pre-registered occupied-cell deep-zone gate. xG is an injected column; silly-kicks ships **no** xG model. Nothing wires into a default VAEP xfn list (opt-in), and no `xthreat` source is modified.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Reuse `scores(window="possession", xg_column=)` as the reward | zero new code | goal-gated (`labels.py:282`) → reproduces the v1 flat deep zone | Factually wrong reward — the degeneracy it must escape |
| B. Widen `xthreat`'s public builders with a `move_actions=` param | DRY | Singh needs all-results, KDE needs success-filtered — one param can't serve both; widens tested API | Two populations; regression surface on classic xT |
| C. (chosen) xtgk-local builders reusing `xthreat` low-level seams; injected `xg_column`; first-shot Markov reward | correct, no `xthreat` edits, parity-gated, decomposable | replicates ~40 LOC of counts | — |

## Consequences

### Positive

- An honest, pressure-conditioned value surface whose deep zone carries a real gradient (validated by the go/no-go gate), unblocking xT-GK v2 sub-projects 2–5.
- `MarkovPossessionValue` reuses `xthreat.value_iteration` verbatim; classic xT stays byte-identical (parity-gated over random cohorts + the frozen oracle).
- `delta_v` exposes a two-factor Shapley split for the metric's decomposition without reaching into estimator internals.
- Pickle-free `save`/`load` (npz + JSON + SHA256) matching the house artifact convention.

### Negative

- `V` is fittable/validatable only where a calibrated per-shot xG exists (the two cohorts) — a deliberate boundary (silly-kicks ships no xG model).
- Two "xt-gk" homes coexist (`tracking/_xt_gk.py` v1 frozen; `xtgk/` v2) until the lakehouse migrates; documented in CLAUDE.md.

### Neutral

- Phase-2 canonical promotion of `V` into the metric, `ρ`, `V_opp`, and the lakehouse migration are separate sub-projects.
- Gate numbers (deep-cell set, effect floor, `N_min`, cross-check tolerance, direction) are owner/Eyestone-locked before fitting; `GateConfig` carries them.

## Amendment (2026-07-09, 4.41.0/PR-S108) — Q3 resolved + G8 frame-aware null-pressure

Two owner-only blockers resolved against the live backend; both refined a design detail (spec rev 4 §5/§6).

- **Q3 — injected `xg_column` = `soccer_analytics.dev_gold.fct_shot_xg.xg`** (calibrated pre-shot, grain `(match_key, action_id)`; a *separate* table from `fct_shot_psxg`, so no post-shot leakage). 100% non-null on both cohorts (WC 1473, RM 2596). **Certification caveat: `ood_flag` = 0 for gradientsports (certified) but 100% for skillcorner (all RM shots OOD).** Per-cohort surfaces (G3) contain it — WC reward clean, RM reward populated-but-uncertified (RM verdict provisional). `MarkovPossessionValue.fit(reward_provenance=)` records a caller-supplied OOD-rate/CI summary (the library never interprets `ood_flag`/CI — no xG model shipped); the owner-run emits `ood_rate_by_source` pre-gate. silly-kicks still ships no xG model.
- **G8 — frame-aware null-pressure rule** (corrects the blanket "fail-loud on missing pressure" in the original §5): distinguish by tracking-frame presence — **frame absent** (genuine gap) → drop/fail-loud (`PressureLevels.apply` backstop); **frame present + `pressure_on_actor` null** (no opponent in the pressure region — a genuinely unpressured restart) → **zero → LOW tercile, keep**. Live: 595/595 GS null-pressure goal-kicks have intact frames; a blanket drop would silently lose 60% of WC goal-kicks (the certified cohort's headline population). Implemented as the pure `coalesce_frame_present_null_pressure(pressure, frame_present)` applied in the owner-run data-prep *before* fit; the unpressured-restart count is reported per cohort (`frame_present_null_pressure_count`) — it is signal, not loss.

No production/xfn change; additive. Phase 11 remains wired-but-not-run, blocked on Q4 (the locked gate numbers) only.
