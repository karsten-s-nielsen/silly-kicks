# ADR-026: Space-creation opponent perspective via complement decomposition on a shared, unmirrored OBSO multiplier

| Field | Value |
|---|---|
| **Date** | 2026-06-11 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen, Claude (luxury-lakehouse mandate) |

## Context

TF-41 (`add_space_creation`, PR-S57/3.21.0) documented a `*_opponent` column triplet that was
hard-coded `np.nan` on every code path since introduction — 100% NULL across all four tracking
providers on 25k+ production rows, while the `_team` triplet was >99.5% populated on the same
rows. 4.22.2 resolved the contract dishonesty by removing the columns; the lakehouse (the
downstream consumer and original reporter) **rejected the removal** and mandated implementation
with exact requirements: same evaluation window/frame/grid as `_team`, directly comparable
magnitudes, `created >= 0` / `destroyed >= 0` / signed `net`, dtype-robust opponent resolution,
a loud failure on frames without exactly two team ids, and an identical NaN mask to `_team`.

The team triplet is the actor's leave-one-out (LOO) differential OBSO on their own team's
attacking surface (Fernandez & Bornn 2018). The opponent triplet must be the same LOO evaluated
on the *opposing team's* OBSO surface — the actor acting as a defender of that surface. Two
design questions follow: how to obtain the opponent surface, and whether the OBSO multiplier
(transition × ball-distance weight × EPV) should be mirrored toward the other goal for the
opponent perspective.

## Decision

The opponent-attacking pitch-control surface is derived as the **exact complement of the same
decomposed baseline** the team side already computes (Spearman: `def/(att+def)`; Fernandez-Bornn:
`sigmoid(def−att)`), so the analytical path costs **zero additional pitch-control computations**;
the Voronoi naive fallback recomputes the opponent surface explicitly per removal. Both
perspectives share the **identical, unmirrored OBSO multiplier**. Corrupt frames (≠ 2 team ids)
**raise `ValueError`** carrying the game/period/frame/action key; NaN actor identifiers keep the
ADR-003 NaN-row default.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Second `compute_space_created` call with `attacking_team_id=opponent` | No primitive changes | The actor is not an attacking-team player of that surface — the LOO iterates attackers only, so the actor gets no row; doubles PC cost; separate call breaks NaN-mask parity | Cannot produce actor-attributed values at all |
| B. Mirror the EPV/transition grids along x for the opponent surface | "Opponent attacks the other goal" intuition | The team-side TF-41 design is direction-agnostic (no `home_team_id`/orientation input; one grid regardless of attack direction) — mirroring would give the opponent triplet an orientation assumption the team triplet never had, breaking the mandated direct comparability and silently changing meaning per frame orientation convention | Inconsistent with the existing team-side semantics it must mirror |
| C. Complement decomposition + shared unmirrored multiplier (chosen) | Zero extra PC cost (analytical); literally identical grid/sigmas/EPV/transition inputs → magnitudes directly comparable; single-call design gives NaN-mask parity by construction | Inherits the team side's direction-agnostic EPV simplification (both perspectives value the same pitch end) | — |

On failure semantics: silently emitting NaN for a resolvable actor on a corrupt (≠2-team) frame
was explicitly forbidden by the consumer — that is exactly the degradation mode that let the
original defect ship for 30+ releases. The guard raises; ADR-003 NaN-tolerance is preserved for
NaN *actor identifiers* (the guard is evaluated only after the actor is resolvable).

## Consequences

### Positive

- The documented contract is honest and live: all six `_SPACE_CREATION_COLUMNS` populate, gated
  by a coverage-parity test (opponent NaN mask == team NaN mask) and a REPO-WIDE liveness gate
  (`tests/tracking/test_aggregator_column_liveness.py`: every registered `add_*`'s added columns
  must be non-null somewhere on a domain-exercising fixture; meta-assertion pins the surface to
  `tracking.__all__`; no exception set) — the original bug class is structurally unshippable for
  ANY aggregator, not just TF-41.
- Defensive value becomes measurable: `space_destroyed_m2_opponent` is opponent OBSO-weighted
  space the actor's presence denies.
- An analytical-vs-naive opponent parity oracle pins the complement decomposition on both
  decomposable methods.

### Negative

- The direction-agnostic EPV simplification is now load-bearing for two perspectives instead of
  one; if TF-41 ever gains orientation-aware EPV, both sides must change together (the shared
  multiplier is single-sourced in `compute_space_created`, so this is one edit site).
- `space_creation_xfns` grew 3 → 6 xfns (18 VAEP columns). It is an opt-in factory in no default
  xfn list; opting in remains a self-triggered VAEP retrain (ADR-005).
- Voronoi (`pitch_control_method="voronoi"`) doubles its per-removal PC cost when the opponent
  perspective is on (explicit opponent-surface recompute; correctness over the complement
  shortcut for the non-decomposable method).

### Neutral

- `compute_space_created` gains `include_opponent_perspective: bool = False`; the default output
  schema is byte-identical to 4.22.1/4.22.2 (additive).
- The two-team guard runs only on the `add_space_creation`/opponent path; team-only primitive
  callers see no behavior change.

## Related

- **Issues / PRs:** lakehouse bug report 2026-06-11 + rejection letter (option 1 mandated); #125 (4.22.2 removal, superseded by this implementation)
- **ADRs:** ADR-003 (NaN-safe enrichment — boundary preserved), ADR-005 (xfns conventions), ADR-019 (dtype-safe id comparisons — `ids_match` opponent resolution)
- **External references:** Fernandez & Bornn (2018), "Wide Open Spaces", MIT Sloan Sports Analytics Conference (see NOTICE)
