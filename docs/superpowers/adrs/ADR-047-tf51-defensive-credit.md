# ADR-047: TF-51 per-event defensive credit/debit family

| Field | Value |
|---|---|
| **Date** | 2026-07-23 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

## Context

silly-kicks values the on-ball event stream (VAEP) and measures pressure (`pressure_on_actor`), but
nothing assigns **signed defensive credit** — the attribution a coach reasons in ("that press earned
the turnover", "that defender got beaten for the chance"). TF-51 adds a per-event defensive
credit/debit family: proximity-gated signed values attributed to individual defenders, sized by the
danger removed or conceded (a shot's **xG**, or the attacker's **xT** at the point of a turnover).

The design was substantially pre-locked by the Soccermatics Pro source spec (§W4, module 16.3) and
refined across two spec-review rounds and two plan-review rounds. It depends on the block-detection
columns shipped in 4.56.0 (ADR-046) and on the fitted-xT surface (`xthreat._physical`, ADR-041) and
injected per-shot xG (`xtgk/_xg_reward` pattern) already in the codebase; it adds no runtime
dependency and ships no xG model (both are injected, fail-loud).

## Decision

Ship TF-51 v1 as a pure `silly_kicks/tracking/defensive_credit/` sub-package with three public entry
points — `compute_defensive_credits` (long-form, one row per (action, credited player, rule)),
`add_defensive_credit` (the per-action defending-team aggregate, the C4 +1 action-coupled aggregator),
and `compute_bravery` (event-only, per-team) — over ten named rules (`DEFENSIVE_CREDIT_RULES` closed
vocabulary) plus a per-team bravery rollup. Turnover rules are sized by **`xT(origin)`** (the published,
validated Bischofberger/Bauer/Baca sizing, arXiv:2606.19931); shot/marking rules by shot xG. It ships
**no `*_xfns` factory** (F4 result-leakage) and **no atomic mirror** in v1.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Denied-ΔxT turnover sizing | intuitive | SPADL failed-pass `end` is the interception point, not intent | B2: mis-sizes; superseded by xT(origin) |
| B. Reverse-xT "position won" sizing | rewards high pressing | diverges from the validated standard; under-values last-ditch defence; measures offensive value | deferred to a v2 pressing lens, not the default |
| C. Individual `cross_block` credit rule in v1 | symmetric with `shot_block` | no clean sizing (no xG, blocked cross has no resulting shot, B2 destination, low crosser-origin xT) | deferred to v2 (folds into the DPA model); v1 counts blocked crosses in team bravery only |
| D. Emit `*_xfns` for VAEP | composes with HybridVAEP | gates on the action's own result + downstream shot outcome → F4 result-leakage | rejected; guarded by an auto-discovering absence test (TF-48 precedent) |
| E. **xT(origin), long-form + defending-scoped aggregate, no xfns, no atomic mirror (chosen)** | validated sizing; honest NaN discipline; no retrain | v1 nearest-defender attribution (no marking model); two-team bravery assumption | — |

## Consequences

### Positive

- Coach-grade per-defender attribution: `compute_defensive_credits` long-form + a per-action
  defending-team aggregate; consumers roll up by `player_id`.
- **Additive → in no default xfn list → no VAEP retrain.** New aggregator only (C4 30 → 31).
- Honest edge handling: rule-fired-but-unsizable → NaN long-form row; rule-did-not-fire → no row
  (ADR-043); the aggregate is always finite. On-/off-target is a tri-state (goal → on-target; injected
  column; else TF-48 `shot_on_target_derived`; unknown → abstain), so a pressured **saved** shot is
  never mis-signed as a positive credit.
- Bravery exposes the v1 set-piece-cross gap (`bravery_set_piece_crosses = NaN` + a faced-count)
  rather than silently dropping it (R2-2), and keeps a usable known-domain headline.

### Negative

- v1 uses nearest-defender attribution (a marking-assignment / role-responsibility model — the Paper 2
  DPA blueprint — is deferred to v2), a lane-free `shot_block` blocker, a ΔxT-threshold through-ball
  test, and a two-team bravery-opponent assumption.
- Box geometry is duplicated a fourth time (`_params.py` 20.16 vs `_ghost_gk` 20.15); a canonical
  `spadlconfig` penalty-area constant is a tracked cross-cutting follow-up (ADR-021).
- The perf budget transitively depends on `add_shot_goalmouth` honoring `links=` (self-checked by the
  single-link call-count test).

### Neutral

- Deferred to a TF-51 v2 (own spec): atomic mirror, individual `cross_block` rule, the reverse-xT
  pressing lens, the DPA/role-responsibility model (arXiv:2606.19931), a lane-geometry blocker, and a
  line-break-gated through-ball.
- The owner-run SkillCorner construct-validity cross-check (`scripts/validate_defensive_credit_vs_skillcorner.py`,
  spec §12) is a **v1** validation follow-up, not a v2 feature — it validates v1's derived pressure/
  beaten/recovery credit against SkillCorner Game-Intelligence native labels (reported-not-gated).
  Shipped as the next branch after this library PR; the library stays provider-agnostic.

## References

- Spec: `docs/superpowers/specs/2026-07-22-tf51-defensive-credit-design.md`
- Plan: `docs/superpowers/plans/2026-07-23-tf51-defensive-credit.md`
- Sumpter, Soccermatics Pro module 16.3; Bischofberger/Bauer/Baca "Blame is easier than praise"
  (arXiv:2606.19931). See `NOTICE`.
- ADR-046 (block-detection prerequisite), ADR-041 (physical xT), ADR-043 (missing ≠ 0),
  ADR-039/042 (F4 xfn leakage), ADR-028 (action-LTR), ADR-019 (id-compat).
