# ADR-046: Block-detection converter columns (`shot_blocked` / `cross_blocked`)

| Field | Value |
|---|---|
| **Date** | 2026-07-22 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen; cross-session review (spec rounds 1–3, plan round 1) |
| **Supersedes / amends** | Extends ADR-045 (reflection-kind registry), ADR-001 (converter identifier conventions — this adds a derived per-shot/cross column, not an identifier override) |
| **Source spec** | `docs/superpowers/specs/2026-07-22-block-detection-converter-columns-design.md` |
| **Source plan** | `docs/superpowers/plans/2026-07-22-block-detection-converter-columns.md` |

## Context

Canonical SPADL has no "blocked" `result_id`: a blocked shot or blocked cross flattens to
`shot`/`cross` + `fail`, indistinguishable from an off-target / saved / overhit / intercepted
action. TF-51 (per-event defensive credit/debit) needs the blocked signal in two places — its
`shot_block` credit rule and its event-only "bravery" metric (% of opposition shots + crosses
blocked). The signal is **present in the raw stream of most providers and dropped during
conversion**: provider audits plus real-data probes (SkillCorner + Gradient Sports via the pining
loader) found `shot_blocked` derivable for 6 of 8 providers and `cross_blocked` for 3 of 8.

This is a **prerequisite** — it ships before TF-51 so TF-51 reads a stable, first-class column
rather than re-deriving the signal per provider. It is purely additive (no existing column or value
changes → no VAEP/tracking retrain) and C4-free (a converter output column, not an action-coupled
aggregator).

## Decision

Add two nullable-boolean columns — `shot_blocked` and `cross_blocked` (pandas `"boolean"` dtype) —
to the shared `SPADL_COLUMNS` schema (which propagates to every provider schema via `**SPADL_COLUMNS`
spread), emitted by **every** converter. A shared `silly_kicks/spadl/utils._blocked_flag(n, *,
applicable, blocked)` helper builds the column with 3-valued semantics: `True`/`False` on
shot/cross rows the provider encodes, **`pd.NA` on non-shot/non-cross rows AND on providers that
cannot encode the signal** (a non-applicable row is "unknown", never `False`). Feasible providers
set the real mask; Opta (unverified qualifier) and SkillCorner (no signal, real-data verified both
tiers) emit all-`pd.NA`; StatsBomb `cross_blocked` is deferred to `pd.NA` (n=1-verified, fragile
`related_events` join). `cross_blocked` is scoped to the open-play `cross` type only; set-piece
`corner_crossed`/`freekick_crossed` stay `pd.NA` in v1.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Provider-specific schema columns (the `is_synthetic` / `result_source` precedent) | matches existing extension pattern; per-provider isolation | not cross-provider-consistent; TF-51 would branch per provider | the contract requires the same column on every provider's output |
| B. Per-converter `extra_columns` append (the `preserve_native` mechanism) | base schema untouched; each converter independent | not canonical/discoverable; dtype not schema-enforced; per-converter empty-path handling | the columns are a first-class documented contract TF-51 depends on, not caller-requested provider fields |
| C. Base `SPADL_COLUMNS` canonical column (chosen) | canonical, dtype-enforced, discoverable, empty-paths auto-covered | couples the schema change to all-converter emission in one commit; two hardcoded count tests + the reflection registry must update | — |

## Consequences

### Positive

- TF-51's `shot_block` rule + bravery read a stable, typed cross-provider signal (no per-provider re-derivation).
- The signal is useful codebase-wide, not just to TF-51.
- Real-data-verified where it matters (GS = the owner's primary data: `shot_outcome_type`/`crossOutcomeType == "B"`, pining-probed).

### Negative

- Every converter must emit both columns (strict-projection `_finalize_output` KeyErrors otherwise), so the schema change + all-8-converter emission land in one commit.
- Blocked-cross coverage is thin: only GS (real) + Wyscout (mechanism-only) populate it; StatsBomb/Opta deferred, DFL/Metrica/kloppy/SkillCorner infeasible → a bravery consumer gets a per-final-action-type NaN "unknown" cross component on those providers, never a fabricated 0.
- Opta and Wyscout are mechanism-only (no committed real fixtures) — recorded as such; a real-data instance probe at their next fixture availability is the standing verification debt.

### Neutral

- Atomic-SPADL is unaffected (it projects to `ATOMIC_SPADL_COLUMNS` and drops the two SPADL columns).
- Both columns are `"invariant"` under coordinate reflection (boolean flags, not geometric) — declared in the ADR-045 reflection registry.
- Column-set tests that compare against `list(X_SPADL_COLUMNS.keys())` auto-adjust; only `test_schema.py` (14→16) and `test_reflection.py` (32→34) hardcode the count.

## Related

- **Specs / plans:** the design + plan above.
- **Downstream:** TF-51 (`docs/superpowers/specs/2026-07-22-tf51-defensive-credit-design.md`) consumes `shot_blocked` (its `shot_block` rule) and both columns (bravery).
- **Verification:** real fixtures — StatsBomb `7298` (12 blocked shots), Sportec IDSSE `per_period` (4, incl. 1 own-team deflection → `False`), Metrica `per_period` (1), kloppy `metrica_events` (3); GS pining-probed (WC2022 match 10502); owner-gated GS e2e recorded as a follow-up.

## Notes

`cross_blocked` was extended into this PR after a real-data GS probe (WC2022 match 10502) showed
`crossOutcomeType == "B"` (⟺ `incompletionReasonType == "BL"`, perfectly aligned, 6/39 crosses) —
the provider audit had reported it "unverified" because the GS *synthetic* test fixture only emits
`{null, C, F}`. Real data was the arbiter, as it was for SkillCorner (whose 294-column Game
Intelligence schema, identical across public and owner-tier RM, records no shot/cross-block signal).

## Amendment (2026-08-19): StatsBomb `cross_blocked` un-deferred

The original decision deferred StatsBomb `cross_blocked` to all-`pd.NA` ("n=1-verified, fragile
`related_events` join"), recording a real-data instance probe as the standing verification debt. That
debt is discharged: a pre-registered probe (spec `2026-08-19-sb360-cross-blocked-and-licensed-coverage`)
over ~510 open-data matches (~10,550 open-play crosses) passed all three rules -- R1 (absent
`related_events`) 0.035, R2 (same-team links) 0.007 -- one same-team case, a StatsBomb offensive
block, correctly excluded -- R3 (multi-block ambiguity) 0, base rate 1.3%.
The `related_events` -> `Block` join is clean and symmetric; the "fragility" was the tiny base rate,
not a broken mechanism.

StatsBomb `cross_blocked` now ships a real mask: an open-play `cross` whose `related_events` links to a
`Block` by the OPPOSING team and NOT flagged `block.offensive`. The single same-team case in the whole
corpus was a StatsBomb-labelled offensive block (Ronaldo, WC2022), which the opposing-team rule already
excludes; the `not block.offensive` guard is explicit belt-and-suspenders. Scope unchanged: open-play
`cross` only (set-piece crosses stay `pd.NA`). Measurement: `docs/research/sb360_cross_blocked/`.

**Hyrum note:** StatsBomb `cross_blocked` flips from all-`pd.NA` to real `True`/`False`. Silly-kicks-side
this is additive and consumed by no `vaep/`/`atomic/` feature -> no retrain; declared `"invariant"`.
(It IS consumed by the public TF-51 `compute_bravery` (not `add_press_commitment`), which has no `*_xfns`:
StatsBomb bravery becomes cross-inclusive rather than shots-only -- a public-function output change,
still no retrain.)
Downstream consumers that added `cross_blocked` to a schema assuming a stable all-`pd.NA` column see a
live-surface value change (see `docs/PRIVATE_CONSUMERS.md`).
