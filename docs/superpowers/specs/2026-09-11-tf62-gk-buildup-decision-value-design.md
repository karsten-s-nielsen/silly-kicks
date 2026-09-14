# TF-62 — GK build-up Decision-quality (`gk_decision`), the xT-GK "Decision Value" extension

| Field | Value |
|---|---|
| **Date** | 2026-09-11 |
| **Status** | DRAFT (for independent review) |
| **Author** | silly-kicks session |
| **Deciders** | Karsten Nielsen |
| **Grounding** | Eyestone collaboration handoff 2026-09-09 (App1 "Decision Value"); owner-run spike 2026-09-11 (below) |
| **Spike evidence** | §3 — responsiveness/instrument-validity spike (owner-tier corpus, aggregate stats only) |

## 0. Phasing (native-first — dependency + validation order)

**ONE cycle, ONE coherent fully-tested commit** (owner approval gate immediately before the commit; no
micro-commit cadence). Native-first is a **BUILD ORDER, not two commits** — Phase 1 (the native anchor)
is built and validated first because the SkillCorner-native tier is the **validation anchor** the
reconstruction fidelity leg compares against, through the **same engine**; Phase 2 (reconstruction) is
built on the same feature branch and both land as one commit. The CODE bundles no owner-run weights (it
reuses the already-bundled `PassCompletionModel`), so there is no weight-provenance split of the code — it
is ONE commit. The construct-validity **report** (`docs/research/gk_decision_construct_validity/`) is a
SEPARATE, provenance-mandated 2nd commit: the research-artifact gate requires `run_tree_dirty: false`, so
the battery re-runs clean on the committed code (the ADR-090 pattern). Two commits total — code, then
report — neither a micro-commit. See ADR-092 (Decision) and §12.7.

- **Phase 1 — the validated native anchor.** `gk_decision` engine + `OptionSet` port + `SkillCornerGIOptionSet`
  (native) adapter + the `providers/skillcorner` GI parse port + `GkDecisionParams`/`GkDecisionReport`/
  `_columns` + the battery legs that need no reconstruction (responsiveness, discrimination, net-of-team,
  transfer). Needs **no** packing/goal-map (native carries `n_opponents_bypassed`). Independently valuable
  (SkillCorner users) and fully validated (the spike's Stage 1).
- **Phase 2 — the reconstruction path (coach/Jeff payoff).** `ReconstructedOptionSet` (SB360 + full-tracking)
  + the reachability filter + the reused `compute_packing_metrics` progression + the promoted public
  `tracking.action_ltr_goal_map` (+ `territorial_defense` migration onto it) + the battery's **fidelity**
  leg (reconstructed vs Phase 1's native, on the Rosetta Stone) + the SB360 responsiveness leg. Validated
  against Phase 1's anchor.

This design doc covers **both** phases; the implementation is split into two plans (Phase 1 first, then
Phase 2), but both phases land as ONE coherent commit (§12.7).

## 1. Motivation — reframe VALUE (quarantined) into DECISION quality (licensed)

xT-GK possession **value** does not support keeper ranking: v1 is near-constant across keepers
(formulation degeneracy, ADR-024), and v2's possession-value surface is **not construct-validated**
(ADR-036's 4.45.0/4.46.0 amendments — outcome-AUC below baselines, keeper-flat/near-zero ICC), ~80 %
team-confounded per the Eyestone collaboration ("ranking not licensed"). So a per-keeper
distribution-**value** metric is not on the table.

TF-62 measures something different: **decision quality** — the keeper's chosen distribution scored
**against the option set they could have played**. Decision Value = `value(chosen) − E[value(available options)]`.
Scoring chosen-vs-available **normalizes out the team-created option set by construction**: a keeper on a
team that manufactures great options is not credited for the options, only for *which one they pick*. This
is the mechanism that can carry a keeper-attributable signal where raw value cannot — and the spike (§3)
confirms it does, at scale.

This is **not** a keeper leaderboard (§8). The deliverable is a **method + calibrated uncertainty + a
multi-provider reconstruction path**. Keeper *ranking* is a separately-gated future step (a crossed
keeper+team ICC / transfer analysis over a larger multi-club transfer corpus), unchanged from ADR-090's
stance for the sister territorial-defense metric.

## 2. Non-goals

- **No keeper ranking / leaderboard** shipped (the transfer identification is underpowered — §3, §8).
- **No hierarchical Bayesian rating in the library** — the rating/ICC/transfer analysis is
  consumer-side + the owner-run construct-validity battery (ADR-009; mirrors how xthreat ships the grid
  but not the archetypes, and territorial_defense ships the arms but the battery lives in `scripts/`).
- **No xG/xPass model shipped as a default** — xPass is the **injected/bundled** `PassCompletionModel`
  (already in `main`); the metric is a `compute_*`, in **no** default xfn list, additive, no VAEP retrain.
- **No event-only-provider support** (Wyscout/Opta): reconstruction needs positions.

## 3. Spike evidence (the gate that licenses this build)

Owner-run 2026-09-11 on the owner-tier SkillCorner GI corpus + the pining SB360 corpus. **All figures are
non-reversible aggregate statistics** (reversibility-not-provenance; raw data never leaves the machine),
the same class the other metrics' construct-validity batteries emit.

**(A) Native mechanism — SkillCorner GI, 906 matches, 22,860 GK decisions, 166 keepers, 92 teams, 17 crossing keepers.**
- **Responsive**: keeper beats a random-choice placebo — `sel_efficiency` 0.700 vs 0.661 (**t = +21.6**);
  `decision_pct` 0.557 vs 0.50 (**t = +23.3**).
- **Discriminating**: one-way keeper ICC 0.029–0.039 vs a keeper-label permutation null (**all p = 0.000**).
- **Net of team**: club-adjusted (leave-one-keeper-out, within multi-keeper teams, 117 keepers) keeper ICC
  **0.09–0.15, p = 0.000**; team-fixed-effect over *all* keepers ≈ 0 (single-keeper teams cannot separate
  keeper from team — the identifiability limit, not absence). Team one-way ICC (~0.03) ≈ keeper one-way.
  **This reproduces Eyestone's "real, modest, largely-but-not-entirely-team" verdict at scale.**
- **Transfer (11 usable crossing keepers)**: **inconclusive** — sign-agreement 45–64% (mixed by metric),
  residual-corr ≈ 0, all n.s. The keeper-*intrinsic* (transfer-robust) component is underpowered/unconfirmed
  → **ranking not licensed** (measured, not assumed).

**(B) Reconstruction fidelity — SkillCorner, 906 matches, 126,488 options.** silly-kicks' bundled
`PassCompletionModel` reproduces SkillCorner's native `xpass_completion`: Spearman ρ **0.746** (completion),
ρ **0.856 / 0.821** (decision-value / selection-efficiency ranking). A ~0.12 conservative mean offset
(WC2022-men's-trained vs the multi-league SkillCorner corpus) → a per-provider recalibration candidate; the
*ranking* (what chosen-vs-available needs) is faithful.

**(C) Zero-velocity SB360 — 321 matches, 1,282 decisions, 227 keepers.** The regime that failed the three
prior counterfactual-defensive metrics (xt-gk-v2, TF-60 Layer-3 arms, TF-54b). Here it does **not** fail —
because the mechanism is different (option-value spread + choice, not pitch-control-surface sensitivity to
one body) — **conditional on the option-set reconstruction using a reachability filter**:
- *Naive* option set (all visible teammates): responsiveness **nil** (`sel_efficiency` t = −0.24) — "all
  visible teammates" over-counts unrealistic upfield options a keeper correctly declines, inverting the signal.
- *Refined* option set (prune to reachable options, xPass ≥ 0.5): responsive (**t = +2.34** sel_eff,
  **+4.33** decision_pct), discriminating (one-way ICC 0.14–0.19, p = 0.000 — team-confounded, fewer
  decisions/keeper inflate it; not directly comparable to (A)).
- Coverage: only ~25 % of GK build-up actions have a usable freeze-frame (SB360 censors deep build-up).

**The load-bearing design consequence:** the `OptionSet` reconstruction **must** apply a reachability /
plausibility filter; "all visible teammates" is measured to break the metric.

Spike scripts are throwaway (scratchpad); they are the seed for the §9 battery, not committed code.

## 4. The metric

Per **GK build-up decision** — a goal-kick or an acting-GK open-play pass (the `gk_distribution_mask`
domain) whose option set has **≥ 3 options** — with option set `O = {chosen} ∪ {alternatives}`:

- **Option value** (pluggable, default = Eyestone's validated form):
  `EV(o) = xpass_completion(o) × (1 + max(0, opponents_bypassed(o)))` — completion × progression, where
  `opponents_bypassed` is the **native** count on the SkillCorner tier and, on the reconstruction tiers,
  the existing **`tracking.compute_packing_metrics(...)["packing_made"]`** kernel evaluated per candidate
  (§5) — **NOT** a bespoke geometric count. This is the single in-library "opponents bypassed" definition
  (glossaried `packing_made`, ADR-039), so no divergent count ships. A backward/lateral option ⇒
  `packing_made = 0` ⇒ progression factor 1 ⇒ `EV = completion` (the boundary the §11 magnitude test pins).
  **Deliberately NOT** `xthreat` (degenerate for keepers, ~0.0005) and **NOT** `passing_option_score`
  (a safety score, ~0-correlated with value). The value function is a frozen-dataclass-parameterised
  strategy (house "pluggable family" idiom, like `xthreat`'s transition family), so an xT-based or
  retention-based value can be added later without touching the engine.
- **decision_value** = `EV(chosen) − mean(EV(O))` (chosen minus mean available; random choice ⇒ ~0).
- **sel_efficiency** = `EV(chosen) / max(EV(O))` (Eyestone's; grand-mean ≈ 0.70 on real data).
- **decision_pct** = fraction of *alternatives* the chosen option beats, **ties split half**
  (`(#EV<chosen + 0.5·#EV==chosen)/#alternatives`), so random choice ⇒ 0.5 exactly even with EV ties
  (the strict-`<` spike proxy is biased low under ties). The strongest discriminator in the spike.
- **n_options**, and a provenance column (§6).

Aggregation to per-`(keeper, match)` (means + counts) is a `summarize_*`; per-keeper ratings, ICC and
transfer analysis are **NOT** in the library (§2, §9).

## 5. Architecture — hexagonal `OptionSet` port + injected xPass + reachability filter

New sibling package `silly_kicks/gk_decision/` (mirrors `gkdv/` · `restdefense/` · `territory/` ·
`territorial_defense/`). The provider-specific complexity (native GI vs positional reconstruction) is
isolated behind ONE port so the decision logic is provider-agnostic.

```
OptionSet (Protocol)                       # provider-agnostic: yields candidate targets per decision
  ├─ SkillCornerGIOptionSet                # native: reads passing_option rows (chosen/received/native EV)
  └─ ReconstructedOptionSet(*, xpass, reachability, ...)   # positional: teammates as candidates + filter
                                           #   SB360 freeze-frames  +  full-tracking frames

option_value(options, *, xpass_model, params) -> EV per option   # pluggable value fn (default comp×prog)
compute_gk_decision_value(actions, *, option_set, xpass, params) -> (samples, GkDecisionReport)
summarize_gk_decision(samples, *, params) -> per-(keeper, match) table
```

- **The reachability filter is a first-class `GkDecisionParams` field** (`reachability_min_xpass`, default
  from the spike; `for_provider` tunable) applied by `ReconstructedOptionSet` — the §3(C) finding, made a
  parameter, not a buried constant. `SkillCornerGIOptionSet` uses the native curated option set and does
  not filter (its options are already genuine).
- **xPass is injected** (`PassCompletionModel`, `.bundled()` default) — fail-closed like every other
  injected model. Per-provider recalibrated variants can bundle later (the §3(B) offset), via
  `from_variant`, no engine change.
- **Chosen identification**: native → the `targeted` option. Reconstruction → the **actual pass**
  (`end_x/end_y`) is the chosen target directly — it is **never snapped** to a teammate (a pass into space
  has no nearby teammate; snapping would fabricate the target). The nearest visible teammate to the pass
  end is excluded from the *alternatives* only, as the presumed receiver, to avoid double-counting; §11
  names a test for a pass-into-space end where that exclusion is ambiguous.
- **Progression via the existing packing kernel (not a bespoke count).** Each candidate's
  `opponents_bypassed` = `compute_packing_metrics(frame, attacking_team_id=keeper_team, goal_map=...,
  passer_xy=keeper_xy, receiver_xy=candidate_xy)["packing_made"]`. Reconstruction consumes action-LTR
  frames (ADR-028; the keeper team attacks +x — verified exact on real SB360: actor position == action
  start, |dx|=|dy|=0.00 m), so `compute_packing_metrics` is passed an **action-LTR `GoalMap`** (the acting
  team's attacked goal = 105, GK-independent) — the **ADR-091 per-action-LTR convention**, NOT a per-match
  `resolve_defended_goals` (which is bimodal on per-action-LTR frames — the exact bug ADR-091 fixed for
  TF-54b). Because the frames are already keeper-attacks-+x, this map is the constant (acting team → 105,
  opponent → 0). Per §12.6 (ADR-055 "ONE implementation of the defended goal end") this is the **promoted
  public `tracking.action_ltr_goal_map` constructor** (PR2), used here **and** by `territorial_defense`
  (its private copy migrated onto it, duplicate deleted) — a single source, never a second copy of the
  orientation logic. Correctness pinned by §11's mirror-invariance gate.

### Package layout (house pattern)
| Module | Contents |
|---|---|
| `_config.py` | frozen `GkDecisionParams` (`min_options=3`, `reachability_min_xpass`, value-fn selector) + `for_provider` (ADR-009) |
| `_report.py` | frozen conserving `GkDecisionReport` (`n_decisions_scored + Σ drop_reasons == n_decisions_in`, ADR-042) |
| `_columns.py` | sample keys/columns + closed provenance vocab (`option_set_source`, drop reasons) |
| `_optionset.py` | `OptionSet` Protocol + `SkillCornerGIOptionSet` + `ReconstructedOptionSet` |
| `_value.py` | `option_value` pluggable family (default completion × progression) |
| `_compute.py` | `compute_gk_decision_value` orchestrator + `summarize_gk_decision` |
| `__init__.py` | public surface |

### Provider tiers (v1 scope — an open decision, §12)
| Tier | Source | Option set | Signal (spike) |
|---|---|---|---|
| **SkillCorner GI (native)** | `passing_option` rows | curated, native EV | strongest (§3A) |
| **SB360 (reconstruction)** | freeze-frame + xPass | teammates, **reachability-filtered** | positive w/ filter, censored coverage (§3C) |
| **Full tracking (reconstruction)** | continuous frames + xPass | teammates, reachability-filtered, velocity-informed | expected strongest reconstruction tier (untested in spike; velocity present) |
| Event-only | — | — | ✗ (no positions) |

The SkillCorner-native tier requires a new `providers/skillcorner/` **GI parse port** (parse
`passing_option` → option rows), mirroring the ADR-084 `appearances.py` pattern: depends on a port +
`id_compat` + pandas only, **never** `silly_kicks.tracking`. The current `spadl/skillcorner.py` discards
the GI layer and is untouched.

## 6. Conservation, honest-NaN, provenance (ADR-042 / ADR-043 / ADR-027 / ADR-055 / ADR-063 / ADR-077)

- Every GK decision is **dropped-and-counted, never a fabricated 0** (the ADR-042 principle): `no_frame`
  (no linked freeze-frame / positions), `too_few_options` (< `min_options` after the reachability filter),
  `unresolved_geometry` (goal end unresolvable — caught at the edge, honest-NaN), `fov_cropped` (SB360
  under-observed local region, ADR-077). `GkDecisionReport` is a frozen conserving Report
  (`n_decisions_scored + Σ drop_reasons == n_decisions_in`) — the ADR-043 `GkdvReport` idiom.
- ids canonical via `id_compat` (ADR-019); keepers grouped on the canonical id, raw id emitted (ADR-085 idiom).
- `option_set_source` provenance over a closed vocab (`native` / `reconstructed`).
- No pitch control ⇒ the ADR-043 `PitchControlCache` landmine does **not** apply; velocity is not required
  for the metric (xPass is positional), so it is Tier-1 on SB360 (the reachability filter is the only
  velocity-independent gate). Full-tracking's velocity is an *enhancement* to the option set, not a
  precondition.

## 7. Dependencies / seams (all already in `main`)

- `expected_passing.PassCompletionModel` (`.bundled()` / `.predict_completion`) — injected xPass. **Why
  this and not the TODO's suggested `ReceiverModel` (ADR-066) / `get_xc` (TF-28):** `PassCompletionModel`
  is the event-only *completion* model the spike validated against SkillCorner native (ρ 0.75); it is
  positional (no velocity), so it works on zero-velocity SB360. `ReceiverModel` predicts the *intended
  receiver* (a different quantity — useful later for chosen-identification on providers without a pass
  end, not for option valuation), and `get_xc` requires the optional `accessible-space` dep **and
  per-player velocity**, absent on SB360 freeze-frames. Both stay reserved, not v1 dependencies.
- `tracking.compute_packing_metrics` — the reused progression kernel (§4/§5). It consumes a `GoalMap`
  (public type): the **continuous/match-LTR** tier uses `tracking.resolve_defended_goals(frames)`; the
  **per-action-LTR SB360** tier uses the **promoted public `tracking.action_ltr_goal_map`** (§5, §12.6 —
  single source, `territorial_defense` migrated onto it).
- `tracking.gk_distribution_mask(actions, frames, resolve_gk="robust")` — the GK build-up domain.
- `keeper_identity.apply_actor_identities_to_frames` — stamps the acting keeper onto anonymous SB360 actor
  rows (ADR-090 built it anticipating "the eyestone GK build-up-decision metric" as its second consumer).
- `providers/statsbomb` + `snapshot_to_tracking_frames` + `load_statsbomb_matches` — SB360 (actions +
  snapshot frames + visible_area).
- The SkillCorner GI parse port (new, §5).
- `xthreat` — only if an xT-based value-fn variant is added (deferred; the default needs no xthreat).

## 8. Honest limits (all reported in the module docstrings, glossary definitions, NOTICE, and the battery report)

1. **Not player-attributable / not a ranking.** The net-of-team keeper signal is real within multi-keeper
   teams (§3A) but the transfer-robust keeper-intrinsic component is **underpowered/unconfirmed** (17
   crossing keepers, **11 usable** at ≥5 decisions/club, n.s.). Keeper ≈ team in magnitude. Ranking is a
   future ADR-009 gated on a crossed
   keeper+team ICC over a larger multi-club transfer corpus. (Same posture as ADR-090.)
2. **Value-function dependence.** The signal is operationalisation-dependent (`decision_pct` strongest;
   `sel_efficiency` weaker). completion × progression penalises deliberately-direct keepers (a long-ball
   team's keeper scores "low") even net of team — "below" ≠ "worse keeper". The pluggable value-fn is the
   hook to explore this; the default is the Eyestone-validated one.
3. **SB360 coverage / censoring.** ~25 % of GK build-ups have a usable freeze-frame; off-frame options are
   invisible. Reconstruction requires the reachability filter (§3C).
4. **xPass calibration offset** across providers (§3B) — ranking-faithful, level-biased; per-provider
   recalibration is a battery follow-up, not a v1 blocker.
5. **Underscored NULL evidence is not "no signal"** — the transfer inconclusiveness and the naive-SB360
   failure are *measured* limits, reported as such.

## 9. Construct-validity battery (owner-run, reported-not-gated; `scripts/validate_gk_decision.py`)

Productionises the spike as an ADR-052 `for_each`-sharded, ADR-037-clean-tree, ADR-056-input-contract
owner-run driver over the owner-tier corpus; emits a `docs/research/gk_decision_construct_validity/` report.
Legs (each pre-registered, aggregate-only):
- **Responsiveness** — chosen vs random-choice placebo (`sel_efficiency`, `decision_pct`).
- **Discrimination** — one-way keeper ICC vs a keeper-label permutation null.
- **Net-of-team** — club-adjusted (leave-one-keeper-out) + team-fixed-effect residual ICC; the crossed
  keeper+team decomposition where the corpus supports it.
- **Transfer** — crossing-keeper residual sign-agreement + correlation.
- **Reconstruction fidelity** — reconstructed vs native decision-value on the SkillCorner Rosetta Stone
  (native GI + tracking on the same match), Spearman.
- **Reachability-filter sweep** — the §3C parameter as a first-class sweep (naive vs filtered).
- **Per-provider xPass recalibration** — the §3B offset.

Promotes **no** default (any promotion is a separate ADR-009). Reproduces the spike's numbers with clean
provenance.

## 10. House-convention compliance

- Hexagonal; `compute_*` **not** `add_*` (C4 action-coupled aggregator count **unchanged**; +1 C4
  container for `gk_decision`, +1 if the SkillCorner GI parse port is a new container).
- In **no** default xfn list; additive; **no VAEP/tracking retrain, no re-materialize**.
- Import-allowlist test both directions (`tests/gk_decision/test_import_allowlist.py`): the package imports
  only public seams (`expected_passing`, `tracking` public, `keeper_identity`, `id_compat`, `spadl`,
  optionally `xthreat`); **nothing imports `gk_decision`** (whole-tree `rglob`); planted-violation meta-tests.
- Frozen `GkDecisionParams` + `for_provider`; conserving `GkDecisionReport`.
- `feature_glossary` entries for every emitted column (ADR-048); `NOTICE` attribution (ADR-005).
- Public-corpus discipline: bundled xPass is public-corpus-trained (`assert_public_corpus`, ADR-038);
  battery numbers are non-reversible aggregates over owner-tier data (reversibility-not-provenance).
- No leakage: `value(chosen)` uses xPass (pre-outcome) + geometry; the metric never reads `result_id` or
  any post-contact outcome ⇒ no `*_xfns` leakage concern (and it ships none).
- **Process (binds the follow-on plan):** the implementation plan carries an explicit
  human-approval-before-commit gate immediately before any commit, and prescribes **one fully-tested,
  coherent commit** (no per-step / micro-commit cadence) — the standing owner rules. Feature branch
  `feat/tf62-gk-buildup-decision-value` (already created), never a worktree.

## 11. Threat model / correctness gates

- **`option_value` MAGNITUDE on known geometry (the load-bearing valuation gate).** A `compute_gk_decision`
  Spearman-ρ reconstruction↔native floor is **monotone-invariant** — a systematically biased bypass count
  (off-by-one, wrong interval bound, wrong band) preserves ranking (ρ stays high) while corrupting every
  absolute `decision_value`/`sel_efficiency`. So a dedicated unit test asserts the **absolute** `option_value`
  on hand-constructed geometry, pinning the bypass **boundaries**: opponent exactly on the pass segment,
  opponent behind the origin, opponent outside the interval, and a **backward/lateral option ⇒
  `packing_made = 0` ⇒ `EV = completion`** (the §4 boundary). Because the bypass term is the reused
  `packing_made`, this also pins that the reconstruction's number equals the glossaried packing definition
  (TF62-SPEC-01) — one geometry, tested.
- **Orientation** — reconstruction must reproject to action-LTR (verified exact on real SB360); a mirror
  invariance test on away-possession decisions, incl. the action-LTR `GoalMap` fed to `compute_packing_metrics`.
- **Reachability filter** — a two-sided gate: with the filter the reconstructed signal is responsive, without
  it it is not (the §3C non-vacuity: a filter that changes nothing would fail).
- **Chosen-snap ambiguity** — a test on a pass whose end lies in space (no visible teammate within a
  threshold of `end`): the chosen target stays the pass end (never snapped), and the receiver-exclusion is
  a documented no-op / flagged, not a wrong alternative drop.
- **Reconstruction ↔ native** — the Rosetta-Stone fidelity test on real frames (ρ floor), a *ranking*
  check that complements (does not replace) the magnitude gate above.
- **Conservation** — `n_scored + Σ drops == n_in` on a multi-domain fixture.
- **id-dtype invariance** (ADR-019) + **NaN-safety** on caller identifier columns.

## 12. Resolved decisions (owner-approved, gold-standard, 2026-09-11)

1. **Package name** — **`gk_decision`** (distinct from `gkdv` = Deterrent Value).
2. **Provider tiers / phasing** — ship **all three tiers, native-first, in ONE cycle / ONE commit**
   (native-first is a BUILD ORDER, not two commits; see §0 and §12.7). The SkillCorner-native tier is
   the validation ANCHOR the reconstruction fidelity leg compares against, through the **same engine** —
   so it is built first, not deferred.
3. **SkillCorner GI parse port** — **yes, a library `providers/skillcorner` GI parse port** (a testable
   provider extractor, ADR-084 `appearances.py` pattern; shaping only — raw GI loading stays scripts-side
   like `_sb_raw.py`), landing in **PR1** with the native adapter.
4. **Default reachability threshold** — **`xpass ≥ 0.85`** (the responsive "reachable = high-completion"
   value), `for_provider`-tunable. The battery's reachability sweep found the metric INVERTS on SB360 at
   0.5 (the bundled WC2022 xPass runs generous there — 95% of options ≥ 0.5, so a 0.5 filter can't bite)
   and is responsive at ~0.85, so 0.5 was a useless default (inert on the native tier, inverting on SB360)
   and 0.85 ships. A deeper per-provider xPass recalibration for SB360 remains a future ADR-009 refinement
   (NOT required for responsiveness — the 0.85 default handles it). See ADR-092 ruling (4).
5. **Value function** — build the **pluggable typed seam + the one validated default**
   (completion × progression); xT-based / retention-based variants are **reserved typed doors, not
   implemented** (the spike proved operationalisation matters, so the seam is load-bearing, not speculative).
6. **Action-LTR `GoalMap` (TF62-SPEC-09)** — **promote ONE public `tracking.action_ltr_goal_map`
   constructor** (ADR-055 "the defended goal end has ONE implementation"); `gk_decision` uses it and
   `territorial_defense`'s private `action_ltr_goal_map` is **migrated onto it (duplicate deleted)**. Lands
   in **Phase 2** (where reconstruction needs it).
7. **ONE cycle, ONE commit (owner decision, 2026-09-12).** Native-first (§0/§12.2) is a BUILD ORDER, not
   two commits: Phase 1 (native anchor) and Phase 2 (reconstruction) land as a **single coherent
   fully-tested commit** on the `feat/tf62-gk-buildup-decision-value` branch, with an explicit
   human-approval gate immediately before it and NO micro-commit cadence. Rationale: the CODE bundles no
   owner-run weights (it reuses the already-bundled `PassCompletionModel`), so Phase 1 and Phase 2 are ONE
   code commit — committing Phase 1 separately would risk exactly the micro-commit churn the owner rejects
   if Phase 2 (which shares the engine + `OptionSet` port and is the zero-velocity SB360 regime that
   `instrument_void`'d three prior metrics) forced a Phase-1 change. The construct-validity **report** is
   nonetheless a SEPARATE 2nd commit: committed research artifacts must carry `run_tree_dirty: false`, so
   the battery re-runs clean on the committed code (ADR-090 pattern) — **two commits total (code, then
   report), neither a micro-commit**. This supersedes the "two fully-validated PRs, each one commit"
   language of earlier drafts (recorded here so the design-of-record cannot later be read as prescribing a
   re-split of the CODE).

## 13. Interfaces (Consumes / Produces — exact signatures for the plan)

```python
# CONSUME (all in main)
PassCompletionModel.bundled().predict_completion(ox, oy, tx, ty) -> np.ndarray
tracking.compute_packing_metrics(frame, *, attacking_team_id, goal_map, passer_xy, receiver_xy, params=None) -> dict  # ["packing_made"] = opponents bypassed
tracking.resolve_defended_goals(frames) -> GoalMap   # continuous frames; SB360 uses a per-frame action-LTR GoalMap (ADR-091)
tracking.gk_distribution_mask(actions, frames=None, *, resolve_gk="robust") -> pd.Series  # bool
keeper_identity.apply_actor_identities_to_frames(frames, actions) -> pd.DataFrame
scripts._loader_pining.load_statsbomb_matches(...) -> Iterator[(prov, mid, actions, frames, home, visible_area)]

# PRODUCE — the engine consumes the OptionSet PORT (fully decoupled from provider data); each adapter
# yields a UNIFORM option-rows table (completion + opponents_bypassed + is_chosen per candidate), so the
# scoring is tier-agnostic and xPass lives INSIDE the reconstruction adapter (PR2), not the engine.
class OptionSet(Protocol):
    def option_rows(self) -> pd.DataFrame: ...   # game_id,period_id,decision_id,keeper_id,team_id,is_chosen,completion,opponents_bypassed,option_set_source
gk_decision.SkillCornerGIOptionSet(parsed_options: pd.DataFrame, *, keeper_ids) -> OptionSet   # native (PR1)
# PR2: ReconstructedOptionSet(actions, frames, *, xpass, reachability, goal_map_for) -> OptionSet
gk_decision.compute_gk_decision_value(option_set: OptionSet, *, params: GkDecisionParams = ...) -> tuple[pd.DataFrame, GkDecisionReport]
# samples (per decision): game_id, period_id, decision_id, keeper (canonical), keeper_raw, team_id,
#   decision_value, chosen_ev, best_ev, sel_efficiency, decision_pct, n_options, option_set_source
gk_decision.summarize_gk_decision(samples) -> pd.DataFrame  # per (keeper, game_id)
gk_decision.option_value(rows, *, params) -> pd.Series      # EV = completion*(1+max(0,opponents_bypassed))
# NEW provider seam (native tier, PR1)
providers.skillcorner.parse_passing_options(gi_events, *, game_id) -> pd.DataFrame  # -> SkillCornerGIOptionSet
```

## 14. Attribution (NOTICE)

Eyestone xT-GK collaboration (App1 "Decision Value"); Singh 2018 (xT lineage, for the reserved xT value-fn);
the injected `PassCompletionModel` (silly-kicks, public-corpus). Per ADR-005, an entry in `NOTICE` +
per-feature docstring cross-links.

## 15. Review disposition (R1 — 2026-09-11)

Independent review verdict: REQUEST CHANGES, no blocking defects, "do not redesign the OptionSet/injected-xPass
architecture." Architecture unchanged; the two SHOULD-FIX and all CONSIDER items addressed:

| Finding | Disposition |
|---|---|
| **SHOULD TF62-SPEC-01** — geometric `opponents_bypassed` reinvents `packing` | §4/§5/§7/§11: reconstruction REUSES `compute_packing_metrics["packing_made"]` per candidate (one glossaried definition, ADR-039), fed an **action-LTR `GoalMap`** (ADR-091) — verified `compute_packing_metrics(...,passer_xy,receiver_xy)` exists with that exact contract. |
| **SHOULD TF62-SPEC-02** — valuation core untested (ρ is monotone-invariant) | §11: added a dedicated `option_value` **magnitude** unit test on hand-built geometry — bypass boundaries (on-segment / behind-origin / outside-interval) + backward option ⇒ `EV=completion`; ρ fidelity noted as a ranking complement, not a substitute. |
| CONSIDER — `decision_pct` ties | §4: ties split half, so random ⇒ 0.5 exactly. |
| CONSIDER — chosen-snap into space | §5: chosen = pass end, **never snapped**; §11 names the pass-into-space test. |
| CONSIDER — ADR cites (036 / 042) | §1: v1 degeneracy = ADR-024, v2 not-construct-validated = ADR-036 amendments; §6: drop-and-count principle = ADR-042, conserving-Report idiom = ADR-043. |
| CONSIDER — 17 vs 11 keepers | §3A/§8.1: reconciled (17 crossing / 11 usable at ≥5/club). |
| CONSIDER — ReceiverModel/get_xc vs PassCompletionModel | §7: note added (completion, positional, SB360-viable; the others are different quantities / need velocity). |
| CONSIDER — plan commit gate / no cadence | §10: process bullet binding the follow-on plan. |
| COULD-NOT-VERIFY — spike numbers, GI fields | Inherent to owner-run NDA validation; the one load-bearing consequence (reachability filter) is a tunable param + two-sided gate (§3C/§11), unchanged. |

**R2 (2026-09-11) — APPROVE.** New non-blocking CONSIDER **TF62-SPEC-09** (the action-LTR `GoalMap` the
reused packing kernel needs has no allowed public seam — `territorial_defense.action_ltr_goal_map` is
private + import-banned) — the reviewer offered either a §11-pinned local constant or promoting a public
constructor. **Owner's gold-standard call (2026-09-11): promote ONE public `tracking.action_ltr_goal_map`**
(ADR-055 "ONE implementation of the defended goal end"), used by `gk_decision` + `territorial_defense`
migrated onto it (duplicate deleted); lands PR2. Recorded in §0/§5/§7/§12.6.
