# ADR-115: the geometry-rate-gate exclusion is a declarable reason (numerics conservation)

| Field | Value |
|---|---|
| **Date** | 2026-10-08 |
| **Status** | Accepted |
| **Deciders** | owner (Karsten); implementing session |

## Context

With the ADR-114 fix in place (commit `c769e86`), the authoritative DGX re-run drove D2 and D3 to completion and the numerics no-flip pass's **map** completed over the full corpus. Its **reduce** then refused:

```
refusing a PASS over a subset: 14 excluded key(s) are not DECLARED
(e.g. ['skillcorner__...', ...]). Add them to --declared-excluded with a reason, or re-run so they are scored.
```

Root cause (measured, not inferred):

- The 14 matches are SkillCorner matches excluded at LOAD by the loader's spec-4.4 geometry admission gate (`silly_kicks/tracking/skillcorner.py` `geometry_rate_gate`): a match is excluded when its **ball** or **player off-pitch RATE** exceeds the public-10-calibrated floor (`_BALL_OFF_PITCH_RATE_MAX = 0.0005` at a **10 m** tolerance; `_PLAYER_OFF_PITCH_RATE_MAX = 0.005` at 3 m). The 10 m tolerance means normal out-of-play (throw-ins, corners, a ball metres over the line) never counts — only a ball teleported **>10 m** off the pitch does, i.e. coordinate corruption, not football. The worst clean match scores 0.00000; a catastrophic sign/origin break scores 0.34139. The 14 excluded matches score 0.00054–0.00290 — real, systematic ball-tracking corruption.
- The gate is **uniform**: d1_a, d1_b, d3_metrics and nf_map all excluded the same 14. Legitimate data-quality admission, not a bug.
- Only `validate_coordination_numerics` reduce calls `assert_exclusions_declared` (D1 derive and D3 validate reduce do not). It requires every excluded key to be DECLARED with a reason from the CLOSED vocabulary `COORD_EXCLUSION_REASONS = {no_tracking, empty_after_filter}` — which had **no token for the geometry-gate reason**. So the legitimate exclusions could not be declared, and `numerics_noflip.json` was unproducible.

The defect is the incomplete vocabulary, not the gate and not the conservation guard.

## Decision

Add a `geometry_rate_gate` token to `COORD_EXCLUSION_REASONS` (the vocabulary's docstring already states "the owner ratifies / extends this set"). The authoritative nf_reduce is then run with a `--declared-excluded` input mapping the geometry-gated matches to `geometry_rate_gate`; the reduce records `{key: reason}` inside `numerics_noflip.json`, so the corpus bound is explicit in the artifact.

This keeps all three gold-standard properties: (1) no silent subset — the conservation gate still refuses any UNDECLARED exclusion; (2) the corpus bound is recorded in the artifact; (3) the declaration is a conscious, reasoned operator INPUT, verified against the map's logged exclusions (every one a geometry-gate exclusion), never the run's own `.excluded.json`.

The declared-excluded JSON is a run INPUT (it carries raw match ids, which stay out of version control per the corpus-id redistribution tiers), exactly like the worker slice files. The reason token and mechanism — not the ids — are what this ADR and the code pin.

## Alternatives considered

| Option | Why rejected |
|---|---|
| **B. numerics stops requiring declaration for loader-level admission exclusions** | Weakens the conservation guarantee (reopens the silent-subset / R2-1 path the gate exists to close) and needs the loader to tag each exclusion with a structured admission-vs-failure category the reduce trusts — more machinery, built to re-introduce an auto-accept path. Wrong direction. |
| **C. relax the 0.0005 / 10 m geometry threshold to admit the 14** | Changes a documented coordinate-corruption floor and would feed corrupted ball coordinates into the no-flip orientation reference — the check most sensitive to exactly that — and forces re-running D1/D2/D3. |
| **re-run so they are scored** | Impossible: the loader refuses them at admission; they cannot be scored. |

## Consequences

### Positive
- `numerics_noflip.json` is producible; the 14 geometry-gated exclusions are declared with a precise reason and recorded in the artifact.
- The conservation guard is preserved intact (still refuses any undeclared exclusion).

### Driver-asymmetry ruling (owner decision, 2026-10-08)
`assert_exclusions_declared` is enforced only by the numerics reduce (`validate_coordination_numerics.py:276`); D1 derive and D3 validate reduce do not enforce it. The asymmetry was escalated to the owner (impl-review D2-REL-02) rather than decided by the session. Measured correction to the escalation's framing: D1/D3 do NOT silently drop the bound — `derivation.json` and `metrics.json` each record `excluded_keys` + `n_excluded` in their `population` block (verified). So the corpus bound IS recorded in every TF-58 artifact; the params are correct (the 14 are legitimately coordinate-corrupted and uniformly excluded). The only gap is uniform ENFORCEMENT + a per-key vocabulary reason across D1/D3, which numerics additionally has.

**Ruling (owner, Y):** the asymmetry is RATIFIED as acceptable for the TF-58 ship — the bound is recorded and the params are correct. Raising D1/D3 to the same enforce+reason bar is an OWNER-APPROVED future item (it would add a reasons field to `derivation.json`, so it is not byte-identical and is deliberately deferred to a later cycle, not folded in here to preserve the D1 reuse). This is an owner decision, not a session deferral.

### Watch
- The declared-excluded list is coupled to the gate's current output: if a match's off-pitch rate later crosses the floor, the list goes stale and the reduce refuses until re-declared — which is the intended conscious-acknowledgement behaviour, not a regression.

Related: ADR-114 (D2 reliability NA-entity parity), ADR-111 (TF-58). Gate source: `silly_kicks/tracking/skillcorner.py` `geometry_rate_gate` (spec 4.4). Vocabulary: `scripts/_coordination_corpus.py` `COORD_EXCLUSION_REASONS`.
