# How-to: construct-validity gates for a metric

> Class-2 procedural runbook (`docs/howto`, sibling to `docs/context`). This centralizes the
> construct-validity discipline that has been re-explained across ~15 mentions and lives, by example,
> in **7** `docs/research/*validity*` memos. It defines the three gates ONCE, pins the verdict/memo
> shape to a real artifact, and says how to add a `validate_*.py`. It cites the exemplar memos rather
> than restating their numbers.

**What "construct validity" means here.** A metric is construct-valid when it measures the latent
skill it claims to — not team quality, not corpus artifacts, not noise. You establish it by scoring
the metric on the **full owner corpus** (never a shrunk sample — fix the engineering, not the corpus)
and passing three gates.

## The three gates

1. **Responsiveness** — the metric MOVES off its neutral baseline, in the right direction, on real
   data. Evidence: a mean + a t-statistic against the null value (e.g. `decision_pct` vs 0.5,
   `decision_value` vs 0), and, where a design parameter exists, a monotone response to sweeping it
   (the reachability-threshold sweep in `gk_decision_construct_validity`). A metric that cannot be
   distinguished from its baseline measures nothing.

2. **Discrimination** — the metric SEPARATES subjects (keepers / teams / players) beyond chance.
   Evidence: an intra-class correlation (`icc`) compared against a **permutation null** (`null_p95`,
   the 95th percentile of the shuffled-label null) with a p-value — `icc` must clear `null_p95`, not
   merely be non-zero. Report it NET OF TEAM too (`net_of_team`: `club_adjusted` /
   `team_fixed_effect` / `team_one_way`), because a metric that only re-discovers team strength is not
   a player skill. A near-zero `team_fixed_effect` ICC with high p (e.g. `decision_pct` p≈0.318) is
   the honest "does not survive team adjustment" reading — report it, do not hide it.

3. **Predictive / transfer** — the metric predicts out of sample or transfers across a split.
   Evidence: season/transfer sign-agreement + residual correlation (`transfer`), and, where the
   metric reconstructs a native quantity, the reconstruction↔native rank correlation
   (`reconstruction.fidelity.rho` with p, n). Weak transfer on a small n (e.g. 11 transfer keepers)
   is a stated limitation, not a pass.

## The verdict / memo shape (pin to a real `metrics.json`)

The canonical artifact is `docs/research/gk_decision_construct_validity/metrics.json`. There is **no
literal `GO`/`NO-GO` string field** — the memo records the gate STATISTICS plus provenance, and GO/NO-GO
is the documented READING of them. Match this shape:

- **Provenance (mandatory):** `run_commit`, `run_tree_dirty: false` (an artifact from a dirty tree is
  refused — ADR-037), and an `input_contract` block: `{version, driver, metric_columns, <seam ids>,
  digest}` — the digest is what the numbers depend on (ADR-056; see the corpus-drivers runbook).
- **`verdicts`:** the population counts (`n_decisions`, `n_keepers`, `n_teams`, …) then one block per
  gate: `responsiveness` (means + `*_t_vs_*`), `discrimination_one_way` (per metric: `icc`,
  `null_p95`, `p`, `n`), `net_of_team`, `transfer`, and any `reconstruction` block.

**Read the verdict as:** responsiveness `|t|` clears significance, discrimination `icc > null_p95`
with `p` below α AND survives `net_of_team`, transfer sign-agreement/`rho` significant on adequate n.
A gate that does not clear is a NO-GO for the claim it backs — report the number, never round it up to
a pass.

## The caveat pattern — state what the number does NOT measure

Every validity memo names its scope limit explicitly. The exemplar is
`docs/research/pass_risk_calibration/` (README.md + metrics.json): it leads with
"**This measures OVER-PREDICTION (specificity on completed passes), NOT DETECTION (recall)**", keeps
the leakage-free headline (`headline_false_alarm_rate`) DISTINCT from the leakage-inflated
`optimistic` block and the CONTAMINATED `low_control_completion_band` (each carries `contaminated:
true` / `leakage_inflated: true` and a `note`), and records a `scope_note` + a `Limitations` section
(selection bias, control≠completion). The rule: a number you cannot cleanly defend is kept, labelled,
and never conflated with the clean headline — never dropped and never silently promoted.

## How to add a `validate_*.py`

A validity harness is a corpus driver — build it on `scripts/_driver.py`, do not re-solve
resume/cache/provenance. Full mechanics in `docs/howto/corpus-drivers-runbook.md`; the validity-specific
parts:

1. Compute the gate statistics over the FULL corpus (responsiveness / discrimination-vs-null /
   transfer), routing all ids through `id_compat` (never raw `str()`/`==`).
2. Make the observed outcome UNREPRESENTABLE where the harness could otherwise answer the research
   question it is meant to authorise — a power/validation harness takes an `InjectionSpec` recipe, not
   an outcome vector (ADR; `docs/context/corpus-drivers.md`). Count an inestimable replicate
   (`n_degenerate_by_size`), never crash or silently skip it.
3. Land the memo as `docs/research/<topic>/README.md` + `metrics.json` with the provenance block above
   (`run_commit`, `run_tree_dirty: false`, `input_contract.digest`).
4. Add a `tests/scripts/test_*` guard (a CI-safe unit slice; the real-corpus run is opt-in / `e2e`).

## Exemplar memos

- `docs/research/gk_decision_construct_validity/` — the fullest triad (responsiveness + ICC-vs-null +
  net-of-team + transfer + reconstruction fidelity + reachability sweep).
- `docs/research/territorial_defense_construct_validity/` — territorial-defense gate.
- `docs/research/xtgk_v2_construct_validity/` — xT-GK v2 gate.
- `docs/research/pass_risk_calibration/` — the caveat / "what it does NOT measure" pattern.
