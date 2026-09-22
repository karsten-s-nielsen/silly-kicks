# TF-56 — Prescriptive defensive-positioning optimiser + measured positioning-gap

**Status:** DRAFT (for review) · **Version / PR / ADR:** next-free, assigned at commit-prep (this spec carries NO pre-claimed number — see the TF-63 collision lesson) · **Base:** a feature branch off `origin/main`.

## 1. Executive summary

TF-56 adds a new event-free package `silly_kicks/positioning/` with two layers on one engine:

1. **A pure prescriptive solver** — `optimise_positions(frame, *, movable, objective, constraints, optimizer)` searches (simulated annealing) for the best *reachable* defensive shape for a single tracking frame under a caller-supplied, composable objective, subject to per-player time-to-intercept (TTI) reachability. It is the prescriptive sibling of GKDV's descriptive counterfactual: GKDV asks "how much did the actual keeper position matter versus a league-average ghost?"; TF-56 asks "where *should* the defenders have been under objective X, given where they could actually get to?".

2. **A measured metric** — `compute_positioning_gap(frames, *, xt, ...)` scores, per defensive frame, `positioning_gap = threat(actual shape) − threat(reachable optimum) ≥ 0` — how much attacking threat the realized shape failed to suppress versus the best reachable repositioning. Shipped at **frame/team grain** (the defending unit's positioning efficiency), aggregated per `(game_id, team_id)`.

The solver is objective-agnostic; the shipped metric column is frozen to **one** canonical objective (threat suppression via `compute_threat_pc`). Composite/pressure/DAS objectives are fully supported through the `Objective` protocol for exploratory/coach use but do NOT back the glossaried column (a composite weighting IS a tactical choice — kept consumer-side by the repo's raw-primitives convention: "the library ships RAW primitives; composites, archetypes and rankings stay consumer-side", CLAUDE.md Key conventions, tagged ADR-009 — distinct from ADR-009's calibration-harness body, which separately backs the frozen `for_provider`-empty params).

Additive, no VAEP/tracking retrain, no re-materialize. A `compute_*`, not an `add_*` — the action-coupled aggregator count stays 33; +1 C4 container. Two-commit clean-provenance cycle: commit-1 is all code (no bundled weights — the solver is pure); commit-2 is the owner-run construct-validity (predictive) + optimizer-stability report, `training_commit` = the clean commit-1 SHA.

**Honest-limit posture (load-bearing):** the metric ships **only if** its commit-2 construct-validity battery returns GO; a NEGATIVE battery demotes it in the same release (out of `__all__`/glossary, code retained private) — the territorial_defense / TF-60-arms precedent. Per-defender attribution and any ranking are explicitly DEFERRED to a future ADR-009 cycle gated on a crossed defender+team ICC over a multi-club transfer corpus (the same defender-vs-team confound that gates territory/gkdv/gk_decision rankings).

## 2. Motivation

silly-kicks owns every scoring engine a defensive-shape objective needs — pitch control (TF-7 `compute_threat_pc`), pressure (TF-2 `bekkers_pi`), accessible space (TF-28 DAS), TTI (`compute_tti`), and ghost frames (TF-18 / GKDV) — but has no *prescriptive* layer that asks where players should be. TF-56 supplies it as a pure, composable search, and turns the optimum into a measured baseline for how efficiently a defence positioned itself.

"**The objective function is the tactics.**" A pitch-control-only objective frees the striker; adding a pressure term pulls a defender to the ball and can abandon a marked man (pressure averages across the marked men — a live aggregate-averaging artifact in silly-kicks' own pressure/DAS). Capping each agent's marginal contribution fixes it. So counterpress / low-block / man-orientation / half-space philosophies are composable objective functions, not hard-coded — and the shipped *metric* deliberately commits to one neutral, defensible objective rather than baking a philosophy into a number analysts read as objective.

Reference design (verified): databallpy v0.8.0 `optimization/optimize_tracking_frame` (Oonk & Shah, MIT; Analytics-Cup winner) — a weighted list of composable objectives + constraints + a `SimulatedAnnealing` algorithm (`num_iterations` ~2000, `patience` ~200 early-stop, Metropolis accept-worse with decaying probability and decaying perturbation). The threat objective maps directly onto `compute_threat_pc` (already a `frame -> float` after fixing `attacking_team_id`/`xt`/`goal_map`), so the canonical shipped-column objective is a thin wrapper. **The other two objectives are NOT free wrappers** and this is genuinely-new code: `bekkers_pi` is a *method string* on `pressure.py`, not a function; the public pressure entry `add_pressure_on_actor` (`features.py`) is action-level (a 33-count `add_*`, `(actions, frames) -> DataFrame`); `get_das` is team-level over frames-plural. So `PressureObjective` and `DasObjective` require **frame→float adapters** (§5). New code in commit-1 = the SA loop + the `Objective`/`Constraint`/`Optimizer` protocols + the two exploratory objective adapters + the metric orchestration.

## 3. Scope

**In scope (v1, this cycle):**
- The pure single-frame solver + `Objective` / `Constraint` / `Optimizer` protocols + built-ins.
- The measured `positioning_gap` column at **frame/team grain**, on the single canonical threat objective, on **continuous velocity-bearing tracking**.
- The commit-2 predictive construct-validity + optimizer-stability (instrument-validity) report.

**Deferred (registered, not this cycle):**
- **Per-defender attribution / ranking** — team-confounded; a future ADR-009 gated on a crossed defender+team ICC over a multi-club transfer corpus.
- **A composite (threat+pressure) metric column** — reachable immediately for exploration via `WeightedSum`; promoting it to a glossaried column needs its weights frozen (a tactical sign-off) + its own construct-validity battery. Zero engine refactor: it is a new frozen `Objective` instance + a new column beside the threat column.
- **SB360 support** — freeze-frames are FOV-cropped (the whole defending unit is rarely in-FOV under an attack, so a shape optimum over a partial team is biased) and carry no velocity (reachability degrades to a from-rest kinematic that loses momentum). A from-rest `ReachabilityConstraint` variant + an FOV-complete-frame gate + its own validation is a future extension. Velocity-less frames are excluded-and-counted, never fabricated.

**Non-goals:** the TIV/archetype/ranking composites (ADR-009 consumer-side); a smooth/gradient optimiser; changing any existing engine.

## 4. Architecture — module + hexagonal boundaries

New package `silly_kicks/positioning/`, a tracking-consuming sibling of `gkdv` / `territorial_defense`:

| File | Responsibility |
|---|---|
| `_config.py` | Frozen `PositioningParams`, `ReachabilityParams`, `SAParams` (`for_provider` empty, ADR-009). |
| `_objectives.py` | `Objective` protocol + `ThreatObjective` / `PressureObjective` / `DasObjective` / `WeightedSum` / `CappedContribution`. |
| `_constraints.py` | `Constraint` protocol + `ReachabilityConstraint`. |
| `_optimizer.py` | `Optimizer` protocol + `SimulatedAnnealing` + `OptimizeResult`. |
| `_solve.py` | `optimise_positions`. |
| `_compute.py` | `compute_positioning_gap` + `summarize_positioning_gap`. |
| `_report.py` | `PositioningReport` (conservation census). |
| `_probe.py` | Commit-2 optimizer-stability + predictive construct-validity battery. |
| `__init__.py` | Public surface (§10). |

**Boundaries:** imports `silly_kicks.tracking` PUBLIC seams + `silly_kicks.id_compat` / `silly_kicks.reflection` ONLY — never a `tracking._*` private (the gkdv/territorial_defense allowlist shape). `xt` is INJECTED (the port pattern; `positioning` never imports `xthreat` for weights). **Nothing imports `positioning`.** An AST import-allowlist gate enforces both directions (`tests/positioning/test_import_allowlist.py`).

A `compute_*`, NOT an `add_*` → action-coupled aggregator count **stays 33**; +1 C4 container. In no default xfn list; **no `*_xfns`** (frame-grain, reads no gamestate/outcome).

## 5. Core solver + protocols

- **`Objective` protocol:** `score(frame: pd.DataFrame) -> float`, lower = better (defensively safer). Built-ins:
  - `ThreatObjective(*, xt, goal_map, params=None)` wraps `compute_threat_pc(frame, attacking_team_id=<opponent>, xt=, goal_map=, method="spearman")` — GK-aware via `lambda_gk`. **Computes the surface directly, NEVER `PitchControlCache`** (ADR-043: the cache key excludes player positions, so a moved-defender frame carrying its twin's identity would be served the factual surface and every delta would collapse to exactly 0 with no warning).
  - `PressureObjective(*, method="bekkers_pi", params=None)` — a NEW `frame -> float` adapter (not a free wrapper: `bekkers_pi` is a method string, `add_pressure_on_actor` is action-level and vectorized). It calls a new public **`pressure_on_target(frame, player_id, *, method, params) -> float`** primitive that REUSES the existing single-method `pressure_on_actor` computer via a synthesized one-row-action call on the `{frame}` set — so **`add_pressure_on_actor` is UNTOUCHED** (no delegation refactor, no new parity golden; the existing `tests/tracking/test_pressure_*.py` remain the guard) and the pressure math is neither duplicated nor restructured. `score` returns `-pressure_on_target(carrier)` (higher pressure = defensively safer → lower score, the shared "lower = better" contract). `DasObjective` — a NEW `frame -> float` adapter wrapping a single-frame `get_das` and reducing to the defending team's conceded-DAS scalar. Both are protocol members / exploratory (the composite-C path), NOT the shipped column.
  - `WeightedSum(list[tuple[Objective, float]])` — itself an `Objective` (composable, no God-object).
  - `CappedContribution(objective, *, cap)` — caps any single agent's marginal contribution to the score (the "abandons a man" aggregate-averaging fix); itself an `Objective`.
- **`Constraint` protocol:** `is_feasible(player_id, candidate_xy, frame) -> bool`. `ReachabilityConstraint(params: ReachabilityParams)` = `compute_tti(real_pos, real_vel, candidate) ≤ max_reach_seconds`, evaluated against the player's REAL pos+velocity (never a mid-search intermediate). Frozen default `ReachabilityParams(max_reach_seconds=0.7, reaction_time=0.1, max_acceleration=SpearmanParams.max_acceleration)` (0.7 s / 0.1 s is the "where should they have been" demo horizon — tighter than databallpy's loose 1.0 s so the optimum stays realistically reachable). An infeasible candidate is rejected at the SA proposal step (never scored).
- **`Optimizer` protocol:** `optimize(frame, *, movable, objective, constraints) -> OptimizeResult`. `OptimizeResult(best_frame, best_score, actual_score, n_iter, n_feasible_proposals, converged)`. Built-in `SimulatedAnnealing(*, num_iterations=2000, patience=200, seed)`: perturb ONE movable player per step (Gaussian, decaying σ), Metropolis accept with decaying temperature, patience early-stop. **The actual shape IS the iteration-0 incumbent**: `actual_score` is that incumbent-0 evaluation (the SAME code path as `best_score`), and `best_score ≤ actual_score` by construction. The metric takes `threat_actual = opt.actual_score` (NOT a second `ThreatObjective.score` call), so `gap = actual_score − best_score ≥ 0` holds by **path identity**, not by comparing two independent evaluations.
- **`optimise_positions(frame, *, movable, objective, constraints, optimizer=None)`** — pure; NEVER mutates `frame`; default optimizer `SimulatedAnnealing()`. **Determinism:** the metric layer seeds the RNG from `(game_id, period_id, frame_id)` so `positioning_gap` is a pure function of inputs (CI-golden-able). A stochastic scored column is unacceptable.

## 6. Measured metric compute + data flow

**`compute_positioning_gap(frames, *, xt, movable=None, objective=None, params) -> (samples, PositioningReport)`.** Per domain frame:

1. **Domain gate:** alive ball ∧ opponent-in-possession ∧ ball within `domain_ball_to_goal_m` of the DEFENDED goal ∧ two-team frame ∧ velocity present ∧ ≥1 movable defender. `domain_ball_to_goal_m` is a frozen `PositioningParams` field whose default reuses GKDV's established danger-domain distance (the plan reads the exact value from `gkdv` at implementation, not a re-invented constant). Out-of-domain → excluded-and-counted. Within domain, sample at a configurable rate (default **1 fps** — 25 fps is massively autocorrelated).
2. **Orientation:** `goal_map = resolve_defended_goals(frames)` (ADR-055, built ONCE and threaded); the defending team's attacked goal fixes `attacking_team_id` (the opponent) for `compute_threat_pc`. Never team-identity.
3. `opt = optimise_positions(frame, movable=<defending outfielders>, objective=ThreatObjective, constraints=[ReachabilityConstraint])` — ONE call yields both `opt.actual_score` (the incumbent-0 evaluation) and `opt.best_score`.
4. `threat_actual = opt.actual_score`; `threat_optimum = opt.best_score`.
5. `positioning_gap = threat_actual − threat_optimum ≥ 0` (higher = worse-positioned; ≥ 0 by path identity, §5).

**Movable default** = all outfield defenders of the defending team (GK EXCLUDED from `movable` — keeper positioning is GKDV's domain; the GK stays a FIXED agent contributing to pitch control via `lambda_gk`). Caller may override `movable`.

**Samples columns** (per sampled domain frame): `game_id`, `period_id`, `frame_id`, `team_id` (defending; canonical-grouped, raw emitted per ADR-019), `positioning_gap`, `threat_actual`, `threat_optimum`, `n_movable`, `n_feasible_proposals`, `sa_converged`, `positioning_gap_source` ∈ `{scored, excluded_out_of_domain, excluded_not_two_teams, velocity_unavailable, unresolved_geometry, degenerate_no_movable}`. The domain gate is evaluated **per-condition** so the census attributes each drop to its specific reason (a non-two-team frame is NOT folded into `excluded_out_of_domain`), and `PositioningReport` conserves over the per-reason counts.

**`summarize_positioning_gap(samples) -> per-(game_id, team_id)`:** mean gap over scored frames + `n_scored` + counts.

**Conservation (ADR-042):** `PositioningReport` conserves `n_frames_scored + Σ drop_reasons == n_frames_in`. Honest-NaN — never a fabricated 0.

## 7. Edge cases + error handling

- **Velocity-less (declared, e.g. SB360)** → `velocity_unavailable`, excluded-and-counted (Tier-3, ADR-063; the blocker is FOV partial-observation of the shape + from-rest reachability, §3). **Undeclared-missing `vx`/`vy`** → RAISE (caller bug, fail-loud, never swallowed into all-NaN).
- **Unresolved goal geometry** → `GoalEndUnresolvedError` caught at the compute edge → `unresolved_geometry` honest-NaN row, counted (ADR-055 policy-at-edge).
- **Degenerate search** — 0 movable, or every movable pinned (0 feasible candidates) → `degenerate_no_movable` NaN row, counted. Never a fabricated 0-gap.
- **Non-two-team frame** → `excluded_not_two_teams`, counted (a distinct census reason, not folded into `excluded_out_of_domain`).
- **`PitchControlCache` refused by construction** (`ThreatObjective` computes directly) — a `positioning`-level non-vacuity test asserts a moved-defender frame yields a DIFFERENT threat than factual (the ADR-043 "counterfactual measurably differs from its twin" guard).
- **`gap ≥ 0`** — guaranteed structurally by seeding SA with the actual shape as incumbent; asserted on a fixture + a mutation that would break it.
- **Determinism** — same inputs+seed → byte-identical `positioning_gap`; `sa_converged=False` is recorded, not raised (a non-converged frame still yields its best-found gap).
- **Purity (ADR-033)** — `optimise_positions` / `compute_positioning_gap` never mutate input frames.

## 8. Probes + commit-2 validation (GO / NO-GO)

> **AMENDED 2026-09-22 (owner-ratified; supersedes the original dose-based §8).** A commit-1 spike MEASURED the original "impose a dose → `positioning_gap` must rise" instrument to be invalid: `positioning_gap` is anchored to each defender's REACHABLE set, so ANY displacement (of the actual shape OR of the optimum) shifts that reachable set, making the gap **non-monotonic in the dose** — dose-actual `|Δgap|` median **0.73 < placebo p95 1.57** on the fixture, and dose-off-optimum FALLS with δ. The gap is also **orthogonal to concurrent shape-badness** (`corr(threat_actual, gap) = −0.20`) — by construction it measures RECOVERABLE improvement, not how bad the shape is. The dose/responsiveness instrument is therefore RETIRED and replaced with **optimizer-stability + discrimination** (instrument) and **predictive** (construct). The same spike set `SAParams.init_sigma_m` **5.0 → 2.0** (matched to the ~2 m reachable radius): at 2.0 the optimum is **SEED-INVARIANT** (gap std **0.000** over 8 seeds) vs std **0.324** at 5.0 — at the old default the column was partly a seed artifact. Feasibility is 80–96 % at every sigma, so the change costs no per-frame budget.

`positioning/_probe.py` provides the pooled-corpus verdict machinery:

**Instrument validity (cheap; synthetic / fixture, no corpus):**
- **Optimizer stability:** the SA optimum is SEED- and ITERATION-invariant — per-frame gap std across a seed set ≈ 0, and doubling `num_iterations` leaves the optimum unchanged. This is what makes `positioning_gap` a pure function of the shape rather than of the (frame-keyed) seed; it REQUIRES `init_sigma_m` matched to the reachable radius (measured: 2.0 m). VOID iff the optimum is seed-unstable.
- **Discrimination:** gap non-degenerate across a shape spectrum (std > 0; not all ~0 / all-pinned).

**Construct / criterion validity (DGX corpus — the ship gate):**
- **Predictive:** a higher `positioning_gap` at frame *t* predicts MORE conceded threat over the next window (the defence left recoverable suppression on the table → the attack exploits it). This is the ONLY meaningful construct test PRECISELY because the gap is orthogonal to concurrent threat: a positive, significant `corr(positioning_gap_t, conceded_threat_{t+Δ})` is what shows the metric MEANS something. Conceded threat is derived from the corpus (the attacking team's realized action-threat / shots in the window; no external xG needed). Reported with effect size + significance; **GO iff positive ∧ significant**.
- **Discrimination on real data:** non-degenerate, sensible distribution across `(game, team)`.

**Reported, not gated:** search-sanity (`n_feasible_proposals > 0` fraction + `sa_converged` rate); the averaging-artifact demonstration (WITHOUT `CappedContribution` a threat+pressure `WeightedSum` optimum abandons a marked man; the cap fixes it); horizon-sensitivity (the gap SCALE rides on `max_reach_seconds`; report the distribution at `∈ {0.5, 0.7, 1.0}` — evidence the metric is not an artifact of one horizon). `arm_unscoreable` (thin domain / velocity-less) stays a first-class distinct verdict.

**Commit-2 run** on a real velocity-bearing corpus (SkillCorner / Sportec / Gradient Sports) via a `scripts/` driver (ADR-052 shards, ADR-037 clean-tree, `require_clean_tree` + provenance stamping, registered in `tests/scripts/test_provenance_wiring.py`). **GO** (predictive-positive-and-significant ∧ discriminating ∧ optimizer-stable/non-degenerate) → the glossaried column stays. **NO-GO** → **demote** in the same release (out of `__all__`/glossary; solver + metric code retained as private modules; report explains) — territorial_defense / TF-60-arms precedent. Applied numbers are reported-not-gated; the column's ship-status is gated on the battery.

## 8b. Performance + corpus feasibility

The threat objective recomputes a full pitch-control surface every SA iteration (the `PitchControlCache` is refused by design, §7), so cost is the dominant design risk and must be sized before the plan drives commit-1.

- **Per-frame cost (MEASURED, commit-1).** One `compute_threat_pc` ≈ **~1 ms** (the GKDV threat arm's per-frame cost with the 4.92.0 pitch-control hoists). SA perturbs one player and rescores per iteration, so a scored frame ≈ `iters × ~1 ms`. The commit-1 inner-loop benchmark (`tests/positioning/test_positioning_perf_budget.py`, a `_perf_structural`-style op-count guard) MEASURES **~1.8 s/scored-frame @ 2000 SA iters** — the surface recompute per iteration dominates (the cache is refused by design), so `compute_threat_pc` runs exactly `1 + n_feasible_proposals` times per solve (no quadratic re-score; structurally guarded).
- **Corpus feasibility.** At ~1–2 s/frame × 1 fps domain sampling × a few hundred domain-frames/match → **order 10–30 min/match**. Commit-2 therefore runs a **BOUNDED velocity-bearing tracking corpus** (order dozens of matches — NOT the 3,961-match open-data corpus, which is event-only), sharded via ADR-052 `for_each` (per-match shards, parallel across the DGX), so wall-clock = per-match cost ÷ parallelism. Corpus size is chosen for pooled-probe POWER, not maximal coverage.
- **Perf levers (frozen `SAParams`):** `num_iterations` + `patience` are primary. A **coarse-grid pitch-control SEARCH surface** (full resolution only for the final `threat_actual` / `threat_optimum` scores) is a registered lever, NOT v1 by default.
- **Feasibility gate (surface, don't silently ship):** if the commit-1 benchmark shows the frozen defaults make even the bounded corpus infeasible, the coarse-grid-search lever lands as a **commit-1 addition BEFORE the commit-2 run** — never a deferred follow-up that ships an un-runnable battery.

## 9. Testing (commit-1, fixture-scale)

Determinism golden (byte-identical gap for a seed) · **optimizer seed/iteration-stability** (the instrument-validity property the amended §8 gates on — per-frame gap std ≈ 0 across a seed set, and `num_iterations` doubling leaves the optimum unchanged; requires `SAParams.init_sigma_m = 2.0`, the spike-fixed default) · reachability red-green (both sides of the TTI horizon) · `gap ≥ 0` invariant (+ a breaking mutation) · counterfactual non-vacuity (moved-defender ≠ factual threat; `PitchControlCache` not served) · protocol composability (`WeightedSum` / `CappedContribution` are `Objective`s; the cap unit-test) · conservation on a mixed fixture (in/out-of-domain, velocity-less, unresolved-goal, no-movable) · velocity tiering (declared→counted; undeclared-missing→raises) · orientation goal_map-driven not team-identity (away-defending fixture) · purity (ADR-033) · import-allowlist both directions · id-dtype invariance (ADR-019). No `test_bundled` analogue (no weights); the commit-2 driver has its own test + provenance-wiring registration.

## 10. Public surface + cross-cutting registrations

- **`positioning.__all__`:** `optimise_positions`, `compute_positioning_gap`, `summarize_positioning_gap`, `PositioningReport`, `PositioningParams`, `ReachabilityParams`, `SAParams`, `Objective`, `Constraint`, `Optimizer`, `ThreatObjective`, `PressureObjective`, `DasObjective`, `WeightedSum`, `CappedContribution`, `ReachabilityConstraint`, `SimulatedAnnealing`, `OptimizeResult`.
- **Glossary:** metric columns `positioning_gap` / `threat_actual` / `threat_optimum` → `feature_glossary` (`emitting_module="_compute"`, `higher_is_better=False`); count grows. Provenance/diagnostic columns (`_source`, counts, `sa_converged`) stay unglossaried per convention.
- **`metric_contracts` (ADR-098):** register `POSITIONING_METRIC_COLUMNS` + grain `("game_id", "team_id")` (the `summarize` grain, gk_decision precedent) in `METRIC_CONTRACTS` — a new column-emitting family must register or `tests/test_metric_contracts.py` fails.
- **C4:** +1 container `positioning` (analyst → positioning; positioning → tracking public seams; `xt` injected). Regenerate `docs/c4/architecture.{dsl,html}`; aggregator count stays 33.
- **NOTICE:** Oonk & Shah (databallpy `optimization`, MIT); Le et al. 2017 (ghosting); Spearman pitch control; Bekkers pressure; TTI (Pleuler, Soccer Analytics Handbook). Per-feature docstrings cross-link with "See NOTICE for full bibliographic citations."
- **pyproject:** add a `positioning` ruff per-file-ignore for `N803`/`N806` if uppercase math naming is used in `_optimizer`/`_objectives`.

## 11. Release — 2-commit clean provenance

- **Commit-1** (clean tree, human-approval gate): all code + protocols + solver + compute + probes + docs (this spec + the plan + the ADR) + C4 + glossary + `metric_contracts` + NOTICE + fixture tests. NO bundled weights.
- **Commit-2** (human-approval gate): the owner-run predictive construct-validity + optimizer-stability report → `docs/research/tf56_positioning/`, `training_commit` = the clean commit-1 SHA. GO (predictive-positive-and-significant ∧ discriminating ∧ optimizer-stable) → the glossaried column stays; NO-GO → demote in this commit.

**Number assignment:** version, PR (`PR-Snnn`) and ADR numbers are next-free, resolved at commit-prep against `main`; nothing in this spec, the plan, or the code carries a pre-claimed number (the TF-63 triple-collision lesson). Never `git commit` / `push` / merge / tag without explicit per-action approval.

## 12. Reused-engine signatures (grounding, read from source)

- `compute_threat_pc(frame, *, attacking_team_id, xt, goal_map, method="spearman", params=None, field_weight=None) -> float` (`tracking/_cover_shadows.py`) — computes the surface directly, never via `PitchControlCache`.
- `compute_tti(pos, vel, targets, reaction_time, max_acceleration) -> np.ndarray` (`tracking/pitch_control/_spearman.py`).
- `get_das(...)` / `get_individual_das(...)` (`tracking/_das.py`).
- `add_pressure_on_actor(actions, frames, *, method, params)` is action-level in `tracking/features.py` (a 33-count `add_*`, returns a DataFrame); `bekkers_pi` is a `Method` STRING LITERAL in `tracking/pressure.py` (`Method = Literal["andrienko_oval", "link_zones", "bekkers_pi"]`), NOT a function. `PressureObjective` therefore wraps a NEW `pressure_on_target(frame, player_id, *, method, params)` primitive that REUSES `pressure_on_actor` via a synthesized single-action call (§5), touching neither of these directly. `SpearmanParams.max_acceleration = 7.0`, `lambda_gk = 3.0`.
- `resolve_defended_goals(frames) -> GoalMap`, `GoalEndUnresolvedError` (ADR-055).

## 13. References

- Oonk & Shah — databallpy v0.8.0 `optimization/` (MIT).
- Le, Yue, Carr, Lucey (2017) — data-driven ghosting.
- Spearman (pitch control); Bekkers (pressure); Pleuler, *Soccer Analytics Handbook* (TTI).
