# ADR-092: TF-62 GK build-up Decision-quality metric (`gk_decision`) — chosen-vs-available (native + reconstruction tiers, one cycle)

| Field | Value |
|---|---|
| **Date** | 2026-09-12 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

xT-GK possession **value** does not license keeper ranking: v1 is near-constant across keepers
(formulation degeneracy, ADR-024) and v2's possession-value surface is **not construct-validated**
(ADR-036's 4.45.0/4.46.0 amendments — outcome-AUC below baselines, near-zero keeper ICC), ~80 %
team-confounded per the Eyestone collaboration ("ranking not licensed"). So a per-keeper
distribution-**value** metric is off the table.

TF-62 reframes the quantity. It measures **decision quality** — the keeper's chosen distribution scored
**against the option set they could have played**: `decision_value = value(chosen) − E[value(available)]`.
Scoring chosen-vs-available **normalizes out the team-created option set by construction** — a keeper on a
team that manufactures great options is credited only for *which one they pick*, not for the options
existing. This is the mechanism that can carry a keeper-attributable signal where raw value cannot.

**The forcing constraint** was the standing rule that a defensive counterfactual on the zero-velocity
SB360 freeze-frame regime must prove instrument-validity FIRST: that regime already `instrument_void`'d
three metrics (xtgk-v2, TF-60 Layer-3 arms, TF-54b territorial-defense). An owner-run spike (2026-09-11,
owner-tier corpus, non-reversible aggregate statistics only — reversibility-not-provenance) established GO
before any code: the mechanism is responsive + discriminating + net-of-team real (transfer underpowered →
ranking still not licensed), and the reconstruction path is xPass-faithful — **conditional on a
reachability filter** (the load-bearing finding; "all visible teammates" measurably inverts the signal).

## Decision

Ship a new sibling package `silly_kicks.gk_decision` behind a hexagonal `OptionSet` port, **native-first
as a BUILD ORDER, in ONE cycle / ONE coherent fully-tested commit** (the SkillCorner-native tier is the
validation ANCHOR the reconstruction fidelity leg compares against, through the same engine — so it is
built first, not deferred; there is no provenance split forcing two commits — the metric bundles no
owner-run weights, it reuses the already-bundled `PassCompletionModel`). The cycle has two build phases,
both landing in the single commit (spec §0 / §12.7):

- **Phase 1 — the validated native anchor:** the `gk_decision` engine + `OptionSet` Protocol +
  `SkillCornerGIOptionSet` (native) adapter + a new `providers/skillcorner` GI parse port
  (`parse_passing_options`) + `GkDecisionParams`/`GkDecisionReport`/`_columns`/`_value`/`_compute`, and
  the owner-run battery's native legs. Native carries `opponents_bypassed` directly (no packing/goal-map).
- **Phase 2 — the reconstruction tiers:** `ReconstructedOptionSet` (SB360 + full-tracking) scored in
  action-LTR + the reachability filter + the reused `tracking.compute_packing_metrics` progression + the
  promoted public `tracking.action_ltr_goal_map` constructor (`territorial_defense` migrated onto it,
  duplicate deleted; ADR-055) + the battery's fidelity + SB360-responsiveness legs. The package's
  import-allowlist relaxes to the tracking-CONSUMING-sibling shape (tracking PUBLIC seams allowed, a
  `tracking._*` private banned); `compute_gk_decision_value` auto-pulls the adapter's `no_frame` /
  `fov_cropped` drops so the conserving census stays the true decision population.

The six owner rulings (locked 2026-09-11, spec §12): **(1)** package **`gk_decision`** (distinct from
`gkdv` = Deterrent Value); **(2)** all three tiers, native-first, ONE code commit (+ a provenance-mandated report commit); **(3)** a library
`providers/skillcorner` GI parse port (shaping only; raw GI loading stays scripts-side like `_sb_raw.py`);
**(4)** default `reachability_min_xpass = 0.85` (the responsive value; 0.5 inverts on SB360 so it was a
useless default), `for_provider`-tunable; **(5)** a **pluggable typed value-fn seam + the one validated default**
(`completion × (1 + max(0, opponents_bypassed))`), xT-/retention-based variants reserved-typed-not-built;
**(6)** promote ONE public `tracking.action_ltr_goal_map` (ADR-055 "the defended goal end has ONE
implementation") in PR2.

**Value function (default):** `EV(o) = xpass_completion(o) × (1 + max(0, opponents_bypassed(o)))` —
deliberately **NOT** `xthreat` (degenerate for keepers, ~0.0005) and **NOT** `passing_option_score` (a
safety score, ~0-correlated with value). `decision_pct` (fraction of alternatives the chosen option beats,
**ties split half** → random ⇒ 0.5) is the spike's strongest discriminator; `sel_efficiency = EV(chosen) /
max(EV)` and `decision_value = EV(chosen) − mean(EV)` complete the family. Domain = the
`gk_distribution_mask` GK build-up action (goal-kick or acting-GK open-play pass) with **≥ 3 options**.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Ship the xT-GK **value** metric (v1/v2) | Already built | v1 degenerate (ADR-024); v2 not construct-validated (ADR-036), ~80 % team-confounded | Value does not license ranking; decision-quality normalizes the team option set out by construction |
| B. Value options by `xthreat` or `passing_option_score` | Reuses shipped surfaces | `xthreat` is keeper-degenerate (~0.0005); `passing_option_score` is a *safety* score, ~0-correlated with value | Neither operationalises "did the keeper pick the higher-value reachable option" |
| C. Reconstruction-first (SB360 before native) | Coach payoff first | The reconstruction fidelity leg has nothing to validate against in-cycle | Native is the validation anchor; building it second forfeits the strongest validation |
| D. Native-first `gk_decision` + `OptionSet` port + injected xPass + reachability filter (**chosen**) | Provider-agnostic engine; native validates; xPass fail-closed; reachability made a first-class param | Native tier only in PR1; reconstruction + public goal-map deferred to PR2 | — |

## Consequences

### Positive

- A keeper-attributable **decision-quality** signal on the SkillCorner-native tier, validated at scale by
  the spike (responsive `t = +21.6/+23.3`; one-way ICC 0.029–0.039 vs permutation null `p = 0.000`;
  net-of-team club-adjusted ICC 0.09–0.15 `p = 0.000`).
- A hexagonal `OptionSet` port that isolates provider complexity (native GI vs positional reconstruction)
  behind one uniform option-rows schema, so the scoring engine is tier-agnostic and PR2 adds a tier
  without touching the engine.
- A new testable `providers/skillcorner` GI parse port (`parse_passing_options`), mirroring the ADR-084
  `appearances.py` pattern — shaping only, `id_compat` + pandas, never `silly_kicks.tracking`.

### Negative

- **Not player-attributable / not a ranking.** The net-of-team keeper signal is real within multi-keeper
  teams, but the transfer-robust keeper-intrinsic component is **underpowered/unconfirmed** (17 crossing /
  11 usable keepers, n.s.). Keeper ≈ team in magnitude. Ranking is a future ADR-009 gated on a crossed
  keeper+team ICC over a larger multi-club transfer corpus (same posture as ADR-090).
- **Value-function dependence.** `completion × progression` penalises deliberately-direct keepers even net
  of team ("below" ≠ "worse keeper"); the pluggable seam is the hook to explore this, the default is the
  Eyestone-validated one.
- **Full-tracking reconstruction was UNVALIDATED in the spike** (§3 tested the native tier + zero-velocity
  SB360 only); it is validated in-cycle by the Rosetta-Stone fidelity leg (reconstructed-from-tracking vs
  native-GI through the SAME engine, owner-run — the whole reason for native-first ordering). The
  reconstruction path REQUIRES the reachability filter (the §3C finding, now a first-class `GkDecisionParams`
  field); SB360 coverage is censored (~25 % of GK build-ups have a usable freeze-frame).

### Neutral

- **+1 C4 container** (`gk_decision`); the `providers/skillcorner` GI port is a module in an existing
  package, not a new subpackage, so it adds no container. **+5 feature-glossary columns**
  (`decision_value`/`chosen_ev`/`best_ev`/`sel_efficiency`/`decision_pct`) → glossary count 394 → **399**;
  the C4 "derived feature columns" DSL count moves in lockstep (399).
- **No VAEP/tracking retrain, no re-materialize** — additive; a `compute_*`, NOT an `add_*` (the 33
  action-coupled aggregator count is unchanged); in NO default xfn list.
- No pitch control ⇒ the ADR-043 `PitchControlCache` landmine does not apply; xPass is positional, so the
  metric needs no velocity (Tier-1 on SB360 in PR2 — the reachability filter is the only
  velocity-independent gate).
- The metric never reads `result_id` or any post-contact outcome (value uses pre-outcome xPass +
  geometry), so it ships **no `*_xfns`** and has no leakage concern.

### Construct-validity posture (reported-not-gated)

The battery `scripts/validate_gk_decision.py` (ADR-052 shards, ADR-037 clean-tree, ADR-056 input contract)
**RAN on the DGX** over the owner-tier corpus (aggregate-only; report `docs/research/gk_decision_construct_validity/`,
`metrics.json` + `findings.md`) and **promotes NO default**. Results:

- **Native (full 906-match corpus): VALIDATED as an instrument** — responsive (`decision_pct` t≈+27),
  discriminating (keeper ICC 0.03–0.04, p=0 vs a permutation null), net-of-team real (club-adjusted ICC
  0.07, p=0), transfer inconclusive → **ranking not licensed** (as designed).
- **Reconstruction fidelity** (Rosetta Stone, per `(keeper, game_id)`) at the shipped default: sel-eff ρ
  ≈ 0.24, decision_value ρ ≈ 0.25 (both p<0.001, n=1,764 pairs) — moderate, significant; the engine is native-faithful.
- **SB360 reachability sweep** (a threshold GRID): SB360 inverts at reachability 0.5 (the bundled WC2022
  xPass runs generous on SB360, 95% of options ≥0.5, so a 0.5 filter can't bite) and is **responsive at
  0.85** — so **the shipped default is 0.85** (0.5 was a useless default: inert on the native tier,
  inverting on SB360). A deeper per-provider xPass recalibration for SB360 remains a future **ADR-009**
  refinement (NOT required for responsiveness — the 0.85 default handles it).

Provenance: the committed report carries **clean** provenance (`run_tree_dirty: false`), enforced by
`tests/scripts/test_artifact_provenance_output.py`. Because a clean-tree battery run is only possible once
the code is committed, the report ships in a **second, provenance-mandated commit** (a clean-tree re-run
at the cycle commit), exactly as ADR-090's territorial_defense battery did — it cannot ride in the same
dirty-tree commit as the code. The validation figures above are from that run (the metric code is
behaviour-identical between the dev and clean runs).

## CLAUDE.md Amendment

None required. This ADR is additive: it adds a `gk_decision` durable-contract bullet and does not except any
project-wide rule.

## Related

- **Specs:** `docs/superpowers/specs/2026-09-11-tf62-gk-buildup-decision-value-design.md`
- **Issues / PRs:** one cycle on `feat/tf62-gk-buildup-decision-value` — Phase 1 (native) + Phase 2
  (reconstruction) land as ONE commit (§12.7); no per-phase commit.
- **Plan:** `docs/superpowers/plans/2026-09-12-tf62-gk-buildup-decision-value-phase2.md` (the full cycle —
  Phase 1 native anchor + Phase 2 reconstruction; the separate Phase-1 plan was dropped as superseded).
- **ADRs:** builds on ADR-090 (the `expected_passing.PassCompletionModel` / `keeper_identity.apply_actor_identities_to_frames`
  seams it consumes), ADR-024/ADR-036 (the xT-GK value quarantine it reframes), ADR-084/ADR-078
  (`providers/*` port + bridge pattern), ADR-039 (`packing_made` — the single "opponents bypassed"
  definition PR2 reuses), ADR-055 (the ONE-implementation goal-end rule + ADR-091 per-action-LTR convention
  PR2's public `action_ltr_goal_map` serves), ADR-042/ADR-043 (dropped-and-counted conservation + the
  conserving-Report idiom), ADR-019 (id-dtype), ADR-048 (glossary), ADR-005 (NOTICE), ADR-009
  (library-ships-primitives; ranking/tuning are gated consumer/owner concerns), ADR-038
  (public-corpus discipline for the bundled xPass).
- **External references:** Eyestone xT-GK collaboration (App1 "Decision Value"); the injected
  `PassCompletionModel` (silly-kicks, public-corpus); Singh 2018 (xT lineage, for the reserved xT value-fn).
