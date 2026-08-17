# SkillCorner keeper-origin resolution — real-data validation (Phase A, ADR-024 acceptance)

`manifest_all.json` in this directory. Produced by
`scripts/validate_skillcorner_keeper_origin.py` at `run_commit aa34017`, `run_tree_dirty false`,
over the **full 108-match SkillCorner pining corpus** (6,865 GK-distribution action rows). Confirms
the shipped keeper-origin resolver (ADR-024, `resolve_gk_geometry`) on real data — the outstanding
half of ADR-024 that PR-S104 could not close (it validated the *code path*, not the *corpus*).

## The two rate-gates (ADR-024 S1 + S4)

| gate | rate | reading |
|---|---:|---|
| `offpitch_rate` (S1: a gross off-pitch under a correct transform must fail loud, never silently clamp) | **0.000** | no resolved GK-distribution origin lands gross off-pitch |
| `out_of_region_goalkick_rate` — GATED, on the SHIPPED per-provider distrust (S4) | **0.000** | every goal-kick origin resolves inside the own box — the ADR-024 "~100% own-box" acceptance, corpus-wide |
| `raw_native_goalkick_out_of_region_rate` — DIAGNOSTIC (distrust OFF), reported not gated | **0.502** | half of SkillCorner's *native* goal-kick origins are the broadcast-ball artifact (~14-20 m downfield); the resolver corrects all of them |

The before/after split — raw 50.2% out-of-region → gated 0.0 after the shipped SkillCorner distrust
— is the whole point of `native_origin_is_trusted("skillcorner") == False`.

## `gr_x = origin_x` — the action-LTR convention (do not use the frame goal map)

`resolve_gk_geometry` emits origins in the **SPADL action-LTR frame** (the acting team attacks
x=105 and DEFENDS the goal at x=0), so the goal-relative x of a GK-distribution origin **is
`origin_x`**. The driver does NOT route it through `resolve_defended_goals` (a frame,
home-attacks-right, quantity): for an away-team action the two frames are 105 m apart, so a
frame-goal-map `gr_x` flips to ~100 and every away goal-kick reads out-of-box — the ADR-028
orientation trap. This defect was caught **on real data** (the frame-goal-map form scored 28.6%
own-box against the correct 100%); a single-team synthetic fixture is structurally blind to it,
which is why the committed fixture carries an away-team goal-kick and a non-vacuity guard.

## CI gates (structural, all legs)

`tests/scripts/test_skillcorner_rate_gates_structural.py` asserts each rate is computed / finite /
under a loose ceiling **plus** a both-sides mutation that breaches the ceiling. The tight
corpus-baseline numbers above are the owner-run `@e2e` data-contract, recorded here with provenance.
