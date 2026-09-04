# ADR-086: TF-54 + TF-55 player-quality bundle (territorial dominance + Glicko-2 duel ratings)

| Field | Value |
|---|---|
| **Date** | 2026-09-04 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

silly-kicks is rich in team-shape and goalkeeper metrics but thin on **individual-player quality**. Two Twelve/Soccermatics metrics fill that gap and share one grain — the per-`(player, match)` atom, windowed for aggregation: **TF-54 territorial dominance** (the "Van Dijk" metric, Sumpter module 10.2 — how much threat a defender concedes vs prevents through the ground they patrol) and **TF-55 Glicko-2 duel ratings** (a pairwise skill rating over ground-duel outcomes). The owner chose to ship them as ONE cycle grouped BY that shared grain ("Bundle A"), each a sibling event-only `compute_*` package mirroring the `restdefense/` / `shot_stopping/` idiom, in ONE release (TF-54 in commit 1, TF-55 in commit 2).

Two design questions shaped the work. First, **how to value a defender's territory**: the owner's established pattern (cf. `ExpectedThreat(method=…)`) is a sensible default plus a typed door to alternatives, not a single hard-coded rule. Second, **how to resolve a duel's winner** when only some providers encode it natively (sportec `tackle_winner/loser`) while others must be derived — and what to do with duels that have no clear winner.

## Decision

Ship two new **event-only** packages, **`silly_kicks/territory/`** (TF-54) and **`silly_kicks/duels/`** (TF-55). Both are `compute_*` (NOT `add_*` action-coupled aggregators — C4 count stays 33), mirror the frozen-`Params` / `_columns` schema / `_report` sidecar idiom, import `spadl` + `id_compat` ONLY (never `tracking`; AST import-allowlist gate, nothing imports them), group on the CANONICAL player id and emit the RAW id (ADR-019), drop-and-count honestly (ADR-042), document every metric column in `feature_glossary` + `NOTICE`, and add a C4 container each. Additive — **no VAEP / tracking retrain, no re-materialize.**

1. **TF-54.** A defender's *territory* is the **trimmed convex hull** (`trim_fraction=0.70`) of their own-half defensive-action locations (tackle / interception / clearance). Opponent passes whose **destination** lands inside the hull are valued by an **INJECTED** fitted `ExpectedThreat` at that destination (`values_at_points`, the xT VALUE — NOT `rate`, which is an end−start difference that NaNs failed passes): **conceded** (completed) vs **prevented** (failed). Valuation is a **pluggable `method=` family** — `completed_failed` (default) with a reserved typed `counterfactual` door. Frame reconciliation is library-owned: the hull lives in the defender's action-LTR frame and opponent passes are point-reflected `(105−x, 68−y)` into it for membership (ADR-028). A degenerate hull (< 3 non-collinear actions) yields a `territory_hull_source="degenerate"` row with NaN metrics, counted.
2. **TF-55.** Per-`(player, match)` **Glicko-2** rating / RD / volatility (Glickman: SCALE=173.7178, Illinois volatility update, inactivity RD growth). The **match is the rating period**; matches process in ascending-`game_id` order (a chronology proxy) with ratings carried forward. Because a rating system is long-lived, `initial_ratings` **resumes** a later batch and `DuelRatingReport.final_ratings` (canonical-keyed) is the resume seed — resume-equivalence (`compute(m1+m2)` == two batches threaded) is gated on real data; `window` slices the cumulative trajectory (`as_of_end` / `change`), never additively. Duels are **native** (sportec `tackle_winner/loser`) or, where absent, **derived** from the `tackle` / `take_on` result adjacency (an adjacent cross-team pair within a 5 s window; the winner is whoever's action succeeded); the strategy is chosen at frame-set granularity, never per-duel-guess. **Ground duels only** (SPADL has no aerial type); an **indeterminate** duel (no clear winner) is **EXCLUDED and counted**, never a fabricated winner. A leakage-free pure `update_glicko` primitive (validated against Glickman's published worked example) backs the orchestrator; `extract_duels` is public.
3. **Per-provider params ship EMPTY** (ADR-009) — `TerritoryParams.for_provider` / `DuelRatingParams.for_provider` resolve to the base config until a separate gated apply-PR.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Ship TF-54 and TF-55 in separate cycles | They share the player grain AND the event-only `compute_*` idiom; the owner chose the grain bundle. |
| TF-54: hard-code the (counterfactual) valuation | Forecloses the alternative the owner wanted exposed; the `method=` family mirrors `ExpectedThreat` and reserves `counterfactual` as a typed door. |
| TF-54: value the pass ORIGIN, or use `rate` | The threat that reached/died in the territory is at the pass DESTINATION; `rate` returns an end−start difference and NaNs failed passes (the prevented leg). |
| TF-55: include aerial duels / rate an indeterminate duel as a loss | SPADL encodes no aerial duel type; inventing a winner for an unresolved duel biases every rating — exclude and count instead (ADR-042). |
| TF-55: rating period = a fixed time window | The per-match period is the grain the metric reports at and needs no wall-clock; inactivity RD growth still widens uncertainty across matches. |

## Consequences

### Positive
- Two new individual-player-quality metrics on the shared player grain; a pluggable valuation door for territory; a reusable, leakage-free `update_glicko` Glicko-2 primitive.
- Fully additive: event-only, no retrain, no re-materialize; validated end-to-end on real StatsBomb WC2022 open data.

### Negative
- TF-55's derived path has **partial coverage** — only adjacent `tackle`/`take_on` pairs form a duel (~11/match on StatsBomb WC2022), reported honestly via `DuelExtractReport`; a richer adjacency is a follow-on.
- TF-54's failed-pass SPADL `end` is the death/recovery location, not the intended target, so a pass aimed into the territory but dying outside the hull is not counted — a documented under-count.

### Neutral
- Per-provider `for_provider` maps are empty scaffolding until an ADR-009 apply-gate.
- `game_id` order is a chronology proxy for TF-55's rating periods when no match date is available.
