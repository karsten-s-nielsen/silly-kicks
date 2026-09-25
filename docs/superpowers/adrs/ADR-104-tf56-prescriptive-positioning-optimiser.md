# ADR-104: TF-56 prescriptive defensive-positioning optimiser + measured positioning gap

| Field | Value |
|---|---|
| **Date** | 2026-09-21 |
| **Status** | Proposed |
| **Deciders** | Karsten Nielsen |

> Number assignment: the ADR / version / `PR-Snnn` numbers are next-free, resolved at commit-prep against `main`. Nothing in this ADR, the spec, the plan, or the code carries a pre-claimed number (the TF-63 triple-collision lesson).

> **AMENDED 2026-09-22 (owner-ratified).** A commit-1 spike MEASURED the original dose-based validity gate ("worse position → larger `positioning_gap`") invalid: the gap is anchored to each defender's REACHABLE set, so it is NON-MONOTONIC in any dose (dose-actual `|Δgap|` median 0.73 < placebo p95 1.57 on the fixture) and ORTHOGONAL to concurrent shape-badness (`corr(threat_actual, gap) = −0.20`). The dose/responsiveness instrument is RETIRED and replaced with **optimizer-stability + discrimination** (fixture instrument validity) and **predictive** `corr(positioning_gap_t, conceded_threat_{t+Δ})` (corpus construct validity, the ship gate). The same spike set `SAParams.init_sigma_m` **5.0 → 2.0** (matched to the ~2 m reachable radius): at 2.0 the SA optimum is SEED-INVARIANT (per-frame gap std 0.000 vs 0.324 at 5.0), so the column is a pure function of inputs rather than a partial seed artifact. See spec §8 (amended).

## Context

silly-kicks owns every scoring engine a defensive-shape objective needs — pitch control (TF-7 `compute_threat_pc`), pressure (TF-2 `bekkers_pi`), accessible space (TF-28 DAS), TTI (`compute_tti`) and ghost frames (TF-18 / GKDV) — but no *prescriptive* layer that asks where players should have been. GKDV answers the descriptive counterfactual ("how much did the actual keeper's position matter versus a league-average ghost?"); TF-56 supplies its prescriptive sibling ("where *should* the defenders have been under objective X, given where they could actually get to?") and turns the optimum into a measured baseline for how efficiently a defence positioned itself.

The forcing constraints: the objective function IS the tactics (a pitch-control-only objective frees a striker; a pressure term can abandon a marked man because pressure/DAS average across agents), so philosophies must be composable objective functions, not hard-coded, while the *shipped metric column* commits to one neutral, defensible objective. The threat objective recomputes a full pitch-control surface every SA iteration (the `PitchControlCache` is refused by design — ADR-043), so per-frame cost is the dominant design risk and had to be sized before committing to a corpus battery.

## Decision

Ship a new event-free package `silly_kicks/positioning/`: a pure simulated-annealing solver `optimise_positions(frame, *, movable, objective, constraints, optimizer, seed)` behind composable `Objective` / `Constraint` / `Optimizer` protocols, plus a measured `compute_positioning_gap(frames, *, xt, ...)` at frame/team grain whose one canonical column freezes to threat suppression via `compute_threat_pc`. It is a `compute_*` (not an `add_*`; the action-coupled aggregator count stays 33), `xt` is injected, and it ships no weights. The metric ships only if its commit-2 construct-validity battery returns **GO (predictive-positive-and-significant ∧ discriminating ∧ optimizer-stable/non-degenerate)** — see the 2026-09-22 amendment banner; a NEGATIVE battery demotes it in the same release (out of `__all__`/glossary, code retained private — the territorial_defense / TF-60-arms precedent).

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. A God-object solver with the objective hard-coded to threat | Simplest to write | Bakes one philosophy into a number analysts read as objective; no exploration | Rejected: "the objective is the tactics" — composability is the point (the `Objective` protocol + `WeightedSum` / `CappedContribution`) |
| B. A gradient / analytic optimiser | Faster convergence | The threat objective is a non-smooth Voronoi-partition integral (no usable gradient); reachability is a hard feasibility set | Rejected: SA handles the non-smooth objective + hard constraints directly (databallpy precedent, Oonk & Shah) |
| C. Serve the search through `PitchControlCache` for speed | ~free per-iteration re-scoring | The cache key excludes player positions, so a moved-defender frame carrying its twin's `frame_id` is served the FACTUAL surface and every gap collapses to exactly 0 with no warning (ADR-043) | Rejected on CORRECTNESS: `ThreatObjective` computes the surface DIRECTLY; a non-vacuity test proves a moved defender moves the score |
| D. A composite (threat+pressure) shipped column now | Richer signal | A composite weighting IS a tactical choice; promoting it needs frozen weights + its own battery | Deferred: reachable immediately via `WeightedSum` for exploration; the glossaried column stays neutral (threat-only) per the raw-primitives convention (ADR-009) |
| E. Coarse-grid pitch-control SEARCH surface (v1 default) | ~4x faster SA iterations | Full-res re-scoring of a coarse-grid optimum can exceed the full-res actual → breaks the `gap >= 0` path identity; unnecessary at the measured cost | Rejected as v1 default (kept as a registered lever, see §Notes); the measured ~1.8 s/frame makes the bounded corpus feasible without it |
| F (chosen). Objective-agnostic SA + threat-only shipped column + injected `xt` + demote-if-fail | Composable, correct (path-identity gap≥0), neutral column, honest ship gate | Per-frame cost is real (~1.8 s at 2000 iters) | — |

## Consequences

### Positive

- A pure, composable prescriptive layer: `optimise_positions` under any `Objective`/`Constraint`, deterministic (RNG seeded from `(game_id, period_id, frame_id)` → `positioning_gap` is a pure, CI-golden-able function of inputs).
- `gap >= 0` holds by PATH IDENTITY (`actual_score` is the SA incumbent-0 evaluation, the same code path as `best_score`), not by comparing two independent evaluations.
- A new public `tracking.pressure_on_target(frame, player_id, *, method, params)` primitive that REUSES `pressure_on_actor` via a synthesized one-row action — `add_pressure_on_actor` is untouched (no delegation refactor, no new parity golden).
- Additive: no VAEP/tracking retrain, no re-materialize; +1 C4 container; in no default xfn list; no `*_xfns` (frame-grain, reads no gamestate/outcome).

### Negative

- Per-frame cost is ~1.8 s at the frozen 2000-iteration budget, so the commit-2 battery runs a BOUNDED velocity-bearing tracking corpus (dozens of matches, ADR-052 sharded), never the 3,961-match event-only corpus.
- The metric is REACHABILITY-BOUNDED, which makes the naive instrument-validity dose ("worse position → larger gap") empirically FALSE (spike-MEASURED; see the amendment banner): `positioning_gap` scores the *reachable* improvement available from a shape, so a dose that displaces a defender AWAY from its reachable-improvement zone SHRINKS the recoverable gap (the dosed position carries its own shifted reachable set), and the gap is orthogonal to concurrent shape-badness. So the dose instrument is RETIRED; instrument validity is optimizer-stability (seed/iteration-invariance, requires `init_sigma_m=2.0`) + discrimination, and the construct/ship gate is PREDICTIVE — a higher gap at frame *t* predicts more conceded threat over `t+Δ` (the meaningful test precisely because the gap is orthogonal to concurrent threat).
- The gap SCALE rides on the uncalibrated reachability horizon (`max_reach_seconds=0.7`, `reaction_time=0.1` are intent-set; `max_acceleration=7.0` is the calibrated Spearman constant), so the battery reports the distribution at `max_reach_seconds ∈ {0.5, 0.7, 1.0}`.

### Neutral

- Per-defender attribution / ranking is DEFERRED to a future ADR-009 gated on a crossed defender+team ICC over a multi-club transfer corpus (the same confound that gates territory / gkdv / gk_decision rankings).
- SB360 support is deferred: freeze-frames are FOV-cropped (a shape optimum over a partial team is biased) and carry no velocity (reachability degrades to a from-rest kinematic). Velocity-less frames are excluded-and-counted (Tier-3, ADR-063), never fabricated.

## Related

- **Specs:** `docs/superpowers/specs/2026-09-21-tf56-prescriptive-positioning-optimiser-design.md`
- **Plans:** `docs/superpowers/plans/2026-09-21-tf56-prescriptive-positioning-optimiser.md`
- **ADRs:** ADR-043 (GKDV / PitchControlCache identity-key landmine); ADR-009 (raw primitives + frozen `for_provider`-empty params); ADR-042 (conservation); ADR-055 (`resolve_defended_goals`); ADR-063 (velocity tiering); ADR-019 (id_compat); ADR-098 (metric_contracts); ADR-048 (feature glossary); ADR-005 (attribution).
- **External references:** Oonk & Shah — databallpy v0.8.0 `optimization/` (MIT, Analytics-Cup winner); Le, Yue, Carr, Lucey (2017) data-driven ghosting; Spearman (pitch control); Bekkers (pressing intensity); Pleuler, *Soccer Analytics Handbook* (TTI).

## Notes

**Perf feasibility gate (Task 11, MEASURED not estimated).** A commit-1 structural benchmark (`tests/positioning/test_positioning_perf_budget.py`) measures the per-scored-frame cost BEFORE any corpus run: **~1.8 s per scored frame at 2000 iterations** on the fixture — consistent with the spec's ~1–2 s estimate. At 1 fps domain sampling and a few hundred domain frames per match, this is order 10–30 min/match; a bounded corpus of dozens of matches, ADR-052-sharded across the DGX, is feasible. **Decision: the coarse-grid-search lever is NOT landed** (it is incompatible with the `gap >= 0` path identity unless the gap is reported at the search resolution, and it is unnecessary at the measured cost). The structural guards assert the anti-quadratic invariant directly: `compute_threat_pc` runs exactly `1 + n_feasible_proposals` times per solve (no hidden re-scoring), and the feasibility check (`compute_tti`) is exactly one per iteration (linear in the iteration budget — the `compute_threat_pc` call count is deliberately NOT the linearity proxy, being confounded by the cooling-driven feasibility ramp).

**Determinism seed.** Seeded from `(game_id, period_id, frame_id)` via `hashlib` (process-independent), not Python's salted `hash`, so the column is reproducible across processes.

**Optimizer-stability instrument + the `init_sigma_m` default (amendment, 2026-09-22).** The SA perturbation std `SAParams.init_sigma_m` is **2.0 m**, matched to the ~2 m reachable radius the `ReachabilityConstraint` admits. The spike measured that at the retired 5.0 the optimum is seed-noisy (per-frame gap std 0.324 across 8 seeds) — a proposal std far larger than the feasible radius makes the accepted-move set seed-dependent — while at 2.0 it is seed-invariant (std 0.000); an extreme sigma instead collapses feasibility to a degenerate no-move optimum. `optimizer_stability_verdict` (gated by `tests/positioning/test_probe.py`) asserts the optimum is seed- AND iteration-invariant; the non-vacuity companion FLIPS the verdict to `seed_unstable` at the retired 5.0. This instrument is a CHEAP fixture property (no corpus), so it is not recomputed by the commit-2 driver, whose ship gate is the corpus PREDICTIVE verdict.
