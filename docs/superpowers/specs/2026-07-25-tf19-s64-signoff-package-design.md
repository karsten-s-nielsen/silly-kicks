# TF-19 §6.4 sign-off package — plasmode power, derived constants, verdict/routing split — design

**Date:** 2026-07-25
**Target:** silly-kicks (next-free release — confirm with owner at commit-prep) / PR-S`<NN>` / ADR-037 amendment (or new ADR — decide at commit-prep)
**Status:** design — cross-session review rounds 1 and 2 applied (R1: 1 HIGH, 1 MEDIUM-HIGH, 1 MEDIUM, 3 LOW, all accepted; the HIGH verified one level deeper as F7 and resolved by D5. R2: 2 MEDIUM, 2 LOW-MED, 1 risk-flag, all accepted — incl. a vacuous FIREWALL test caught before planning). Awaiting owner sign-off
**Scope note:** D5 roughly doubled this cycle. The alternative the owner weighed it against — register the anchor rule now, derive `N_min` in PR-3b once Layer 2's design exists — is recorded verbatim in D5's row and stays available as a fallback if the confounder join overruns.
**Source:** TF-19 GKDV cycle (ADR-037); spec `2026-07-12-tf19-gkdv-regate-and-v1-design.md` §6.1 + §6.4; the xS-v2 run result (`docs/research/tf19_pr3b_xs_v2/`, silly-kicks 4.60.0 / PR-S131).
**Scope class:** validation/research machinery — in NO xfn list, no VAEP consumer, no action-coupled aggregator → **C4-free (count stays 32), no retrain trigger**.

---

## 1. Executive summary

TF-19's §6.4 discrimination harness is **unsigned-off**, and the spec forbids writing any of its
constants into code until it is. This cycle makes it signable. Three things were known to block
sign-off; verifying them against code surfaced two more.

1. **§6.4 is stale.** It was written when the attempt arm read flat. Since 4.60.0 the xS arm reads
   **live and dose-responsive** (`pass` / re-gate `joins_with_caveat`; ρ=0.436, p=0.001) while
   xCross stays `gated_clean_fail`. The arms **diverge**, and §6.4's framing does not admit that.
2. **Two registered constants are placeholders** the spec explicitly forbids registering bare:
   `N_min` (decision row 5) and the Layer 3 headroom threshold.
3. **`regate_verdict` hard-codes H1** — a registered disclosure that is still outstanding.
4. **DISCOVERED (§2): §6.1's pre-registered plasmode power analysis was never built**, though §6.1
   ships as the PRIMARY criterion and its own registered constant's docstring promises the curve.
5. **DISCOVERED (§2, F6): the shot arm's `entanglement` was never measured.** 4.60.0's headline
   `joins_with_caveat` rests on a driver DEFAULT that only became decision-relevant when the v2
   probe returned `pass`. The driver that measures it is shipped and unrun.

The package delivers all five. Two load-bearing corrections run through it: **`N_min` and §6.1's
power curve are different estimands** — a spell-level ATT and a keeper-level variance share, so
row 5's `"at ICC 0.015–0.026"` clause is a category error imported from §6.1 — and **the xS arm's
supportive-context input is a registered default masquerading in the record as a banked measurement.**

**A third correction arrived from cross-session review and reshaped the cycle (F7 → D5).** The first
draft powered `N_min` on `shot_arm_config`, which is *the same estimand conflation one level down*:
that design's outcome is a **goal** and its treatment is roughly Layer 2's **outcome**. A power curve
is a function of a design, so `N_min` must be powered on Layer 2's — and Layer 2's covariate-threshold
treatment is not expressible in the shipped builder at all. Layer 2's **design** therefore lands in
this cycle (D5), which makes §5.1's **FIREWALL** the constraint the whole package now rests on: once
the design exists in code, the harness that powers it could also *run* it and answer the H1-vs-H2
question before the sign-off meant to authorise it. It always injects; the observed contrast is never
computed. Enforced by a test.

---

## 2. Verified findings (evidence, not inference)

Each was checked against **code**, not against the spec's own registration — the discipline this
repo's rule names ("quote the assertion body, not its registration") and which the spec itself has
been burned by twice (round 2's N2 found `causal/_occurrence_labels` was a phantom on both counts).

**F1 — the plasmode simulator does not exist.** The derivation duty says *"derive from the plasmode
simulator — **it already generates** real strided frames with injected effects."* The string
`plasmode` occurs in **exactly one file in the repository: the spec itself.** Nothing else does the
job: `tests/causal/_fixtures.py` provides geometry builders (`frow`/`frames`/`spell`/`actions`) with
no effect injection and no power machinery. The parts to *build* one are shipped
(`fit_propensity` → `estimate_att` → `CausalEstimate`; `placebo_shift`'s seeded-replicate loop as
the idiom), but it is new code.

**F2 — §6.1's power analysis is an undelivered PR-3 obligation.** `ICC_ANCHORS = (0.015, 0.020,
0.026)` shipped in `gkdv/_validate.py:21` carrying the docstring *"a power curve is reported at all
three rather than at a midpoint"* — a curve **no code can produce**. `_group_metrics.py` ships
`icc_one_way` + `group_spread` only, though the spec's module concept reads *"`_group_metrics.py` =
domain-free grouped statistics (ICC, spread, permutation band, **power sim**)."* §6.1 registers the
precondition unambiguously: *"the gate is registered only if detection at the anchor (α = 0.05,
match-block null) is ≥ 0.8 — **otherwise floors/sampling are adjusted FIRST**."* The primary
criterion is therefore registered with its power precondition undischarged, and the owner validation
run (PR-4) is explicitly supposed to be preceded by it.

**F3 — row 5 conflates two estimands.** §1.3's anchor is scoped, in its own words, to *"GKDV's **ICC
null-band and min-n**"* — a between-keeper variance share. Row 5 reuses it for **Layer 2 ATT power**,
a mean difference on a binary spell-level outcome. No mapping between them is stated anywhere in the
spec (verified: every `N_min` / `power` occurrence read).

**F4 — the H1 routing is prose-only.** `regate_verdict` (`tracking/_model_eval.py:714`) returns
verdict strings and contains no routing. The hard-coding lives in ADR-037 §4 and in the recorded
`docs/research/tf19_pr2/decision_table.md` (*"Per ADR-037's routing rule, a probe fail is not 'no
signal' — it sends the arm to GK feature engineering"*). No Python change is needed to *remove* a
routing string; a channel must be *added*.

**F5 — `openGoal` is a dimensionless fraction.** `_open_goal_fraction`
(`tracking/_xshot_occurrence.py:43`): *"Unobstructed share of the goal mouth from the ball"*,
returning *"Open fraction in [0, 1]; NaN if the ball position is NaN/behind goal"*, and — the line
that matters — ***"The GK is not passed in (excluded as an occluder)."*** That is §3.1's feature
contract in code, and it is exactly what Layer 3's probe reverses.

**F6 — the shot arm's `entanglement` was never measured, and it silently became load-bearing.**
`regate_verdict` consults `entanglement` **only** when the probe verdict is `pass`
(`_model_eval.py:730`: `return "joins" if entanglement == "clears" else "joins_with_caveat"`).
`scripts/validate_xs_probe.py:81` supplies it as a **function default**, `entanglement="inside_band"`,
with a `--entanglement` override — and the driver's own comment at line 173 records the assumption
that dated it: `# inert unless the probe surprises with 'pass'`. **The v2 probe surprised with
`pass`.** The parameter that was inert became the one that decided the verdict.

The value itself is a carry-forward: PR-2's decision table describes `inside_band` as *"the
registered expected outcome per ADR-037 §2 point 2... carried forward from the ADR-037
registration"* for the **cross** arm, and records the shot row as ***"(not run — PR-3-gated)"***.
There is no `docs/research/xshot_causal/` artifact, and `xcross_causal/report.md` contains no shot
section. Meanwhile **`scripts/validate_xshot_causal.py` is a complete, runnable maintainer driver**
whose docstring states it measures exactly this quantity — *"GK-CONFOUNDER ENTANGLEMENT (the
supportive-context input to `regate_verdict`)"*. It has never been run.

Consequences, in order of importance: (a) 4.60.0's headline **`joins_with_caveat` is
default-derived, not measured** — it is the CONSERVATIVE branch, so nothing shipped is overclaimed,
and a measured `clears` would have produced the *stronger* `joins`; (b) TODO.md, CLAUDE.md and the
4.60.0 narrative describe it as resting on "the banked SHOT causal arm", which does not exist — a
record correction this cycle owes; (c) the driver carries a legitimate REFUSAL path
(`degenerate` below `SHOT_ARM_MIN_CONTROL_CONVERSIONS`), so "run it" may honestly return "refused",
which is a reportable result and not a failure.

**F7 — Layer 2's treatment is not expressible in the shipped builder** (surfaced by cross-session
review round 1, then verified one level deeper than the review went). `_label_treatment`
(`causal/opportunities.py:297-301`) assigns treatment **purely by action-type occurrence**:

```python
ts = _team_period_action_times(actions, gid, per, team, cfg.treatment_type_names)
win = ts[(ts > entry) & (ts <= hi)]
return (1, float(win[0])) if len(win) else (0, None)
```

Layer 2's treatment — *keeper depth at final-third-spell entry, binarised at 16.5 m* — is a
**covariate threshold evaluated at a fixed time**, and no `treatment_type_names` value can express
"the keeper was deep". A second, subtler mismatch: treated spells here take their anchor from the
treatment ACTION (`win[0]`), and the outcome window keys on that anchor; Layer 2 has **no anchor
action**, so entry anchors both arms. (One upside worth recording: that removes the treated-vs-control
time-shift the module docstring flags at `:11-12` as a documented modeling choice for the existing
arms.)

Consequence: the review's recommended fix — *"state Layer 2's `OpportunityConfig` explicitly and
power on that"* — is not a config statement. It requires a new treatment **mechanism**. That is a
change to the builder's treatment semantics, not an additive field with a safe default like D3's,
which is why it is called out as its own decision (D5) rather than folded silently into D3.

---

## 3. Resolved design decisions (owner, 2026-07-25)

| # | Decision | Rationale |
|---|---|---|
| D1 | **Build the plasmode in this cycle**, faithful to the registered duty | Also discharges F2, which gates the primary criterion and PR-4 |
| D2 | Row 5's ATT anchor is **scale-free relative to the outcome base rate** | §1.3's own registered lesson: *"scale-free relative criteria + placebo bands are the honest idiom for small-probability quantities"* |
| D3 | **Pull `OpportunityConfig.outcome_max_distance_m` forward** from PR-3b | Row 5 gates BOTH Layer 2 outcomes; an `N_min` derived only on `Y_attempt` is anti-conservative for `Y_close_attempt`, the outcome row 7 actually fires on |
| D4 | **Run the §3.3 shot-arm causal harness** in this cycle (F6) | The driver is shipped and unrun. Converts 4.60.0's headline from default-derived to measured |
| D5 | **Pull Layer 2's treatment MECHANISM forward** (F7): covariate-threshold treatment axis, entry-anchor rule, `layer2_config()`, and the `Y_attempt` / `Y_close_attempt` labellers | `N_min` gates Layer 2, and a power curve is a function of a DESIGN. Powering on `shot_arm_config` would repeat F3's category error one level down — that design's `Y` is a **goal**, and its `Z` is roughly Layer 2's `Y`. Owner decision after the alternative (register the rule now, derive the value in PR-3b) was put alongside it |

---

## 4. Module homes — resolving a spec self-contradiction

The spec says both *"the power simulator is a pure `_validate` function"* (§6.1) **and** that the
power sim belongs in `_group_metrics.py` (§6.1's module concept). That was coherent when there was
one power question. With two estimands it is not, because their **dependencies differ**. Resolved by
import direction, which is the only criterion that cannot be argued with:

| Piece | Home | Why |
|---|---|---|
| ICC-mode power sim | `silly_kicks/_group_metrics.py` | Domain-free: values, group labels, block labels. Imports nothing from `gkdv/` or `causal/`. Sits beside `icc_one_way`, which it powers. |
| Generic ATT power loop | `silly_kicks/causal/power.py` (**new, public**) | Needs `fit_propensity`/`estimate_att`. A domain-free stats module must not import an inference package; `causal/` is already public (ADR-015 promotion). |
| GKDV-specific injection + registered constants | `silly_kicks/gkdv/_validate.py` | Its stated role — *"registered constants + verdict logic"*. Keeps `gkdv/` on public seams only. |

**Deviation, stated:** §6.1's *"a pure `_validate` function"* is **not** followed literally, because
one home for two dependency sets is wrong. `gkdv/_validate.py` retains the registered constants and
the thin callers; only the reusable machinery moves out. Recorded here so a reader meets the
reasoning rather than an unexplained divergence.

**A documented fence is being taken down, and it is named rather than stepped over.**
`_group_metrics.py:30-32` states verbatim: *"the spec §6.1 module concept also names a permutation
band and a power simulator. Those are **PR-3b** and are deliberately absent here — this module holds
exactly what PR-3 lifted."* §4 puts the ICC power sim in that exact file. The override is justified
by F2 — §6.1's own precondition (*"the gate is registered only if detection at the anchor... is ≥
0.8"*) means the primary criterion cannot be interpreted until the sim exists, so deferring it to
PR-3b left a shipped constant promising a curve nothing could produce. **That docstring is corrected
in the SAME commit that lands the sim**; a module asserting its own contents falsely is worse than
the deferral it was recording. (Chesterton's Fence, as this repo names it: find out why the fence is
there, then prove the reason no longer applies — not remove it quietly.)

`causal/power.py` is a NEW PUBLIC module → it joins `tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES`,
the Examples gate, and the CI doctest surface from day one — the treatment §7 already mandates for
`gkdv/` and `causal/`.

---

## 5. Layer 2's design, and the plasmode simulator

### 5.1 Layer 2's design lands in the builder (D5)

**Treatment axis.** `OpportunityConfig` gains a covariate-threshold treatment alternative to the
action-occurrence one: a covariate name, a threshold, and a direction. When it is set,
`_label_treatment` assigns `Z` by thresholding that covariate **at spell entry**, and the anchor is
the **entry time for BOTH arms** (there is no treatment action to anchor on). When it is unset the
existing action path runs unchanged — `xcross_config(model_metadata)` and
`shot_arm_config(model_metadata)` stay **byte-identical**, guarded by the existing `config=None`
regression check plus explicit per-config identity tests. (Signatures pinned here rather than left as
prose shorthand: `shot_arm_config` takes `model_metadata`, and `build_opportunities` is
`(frames, actions, *, home_team_id, model_metadata, advance_m=..., config=None)` — the review's LOW
#1, and the shorthand most likely to reach a plan as a `TypeError`.)

**The treatment covariate is goal-relative x (depth), NOT radial r.** §6.4 registers the binarisation
at *"goal-relative **x** = 16.5 m — the penalty-area line"*, while the shipped GK block is polar
(`gk_block=("GK_r","GK_theta")`). These agree only on the goal's centre line and diverge off-centre,
so depth is taken as `x = GK_r · cos(GK_theta)`. Stated explicitly because thresholding `GK_r`
directly would silently mis-assign treatment for wide spells while looking entirely reasonable.

**Outcomes.** `Y_attempt` (an attempt occurs in-spell) and `Y_close_attempt` (an attempt whose SPADL
origin lies within `D` = 16.5 m of goal centre) via D3's `outcome_max_distance_m`. `Y_far_attempt :=
Y_attempt ∧ ¬Y_close_attempt` is emitted as the registered PARTITION for §6.4's coherence check —
reported, never gated.

**Confounders.** §6.4's six families. Some are already on the spell row from the xS extractor (ball
r/θ, defender distances); **defensive line height and compactness** (TF-14 `compute_defensive_line`)
and **carrier pressure** (pinned to `bekkers_pi`) are tracking aggregates that must be joined onto
spells, and score differential / time remaining come from match context. This join is the bulk of
D5's real work and is called out here so it is not discovered during planning — it gets its **own
plan task with its own gate**, being the piece most likely to blow the estimate.

**Confounder PROVENANCE is registered, because one source is stale.** Every tracking confounder is
**computed fresh from frames** in this cycle. It must NOT be sourced from
`fct_action_context`: ADR-045 / 4.55.0 fixed `pressure_on_actor__bekkers_pi` (the away-team velocity
re-projection defect — away values changed, home byte-identical) and **the lakehouse
re-materialization of that column is still an open owner action**. A mart-sourced join would silently
hand Layer 2's design pre-fix away-team pressure, in a confounder chosen precisely because it is
load-bearing, and no test in this package would notice. If a mart source is ever adopted, it must
assert a post-4.55.0 vintage.

> **FIREWALL — the single most important constraint in this cycle, and it exists only because D5
> passed.** Once Layer 2's design is expressible in code, the machine that powers it can also *run*
> it. Running it on the observed outcome would produce Layer 2's answer — the H1-vs-H2 decider —
> **before the sign-off that is supposed to authorise it**, from a cycle whose whole premise is
> pre-registration. Registered constraint: the **Layer 2 power harness** always injects, and
> **Layer 2's** observed-outcome ATT is never computed, never logged, and never written to an
> artifact. (§5.5's `validate_xshot_causal.py` computes an observed ATT by design and is outside this
> guard — the qualifier is deliberate.) Enforced by a test whose non-vacuity is itself demonstrated
> (§10): a call-count spy cannot see this defect, so the injected vector carries a provenance stamp
> and the guard is shown RED under a deliberate mutation. Layer 2's real contrast stays unread until
> PR-3b.

### 5.2 Plasmode — shared principles

Plasmode, **not i.i.d.** — the spec's reasoning is retained verbatim: *"An i.i.d. simulation would
inherit none of the clustering and could pass while the real instrument is simultaneously
underpowered and anti-conservative."* Both modes therefore build on **real** data and preserve its
clustering.

### 5.3 ICC mode (§6.1, discharges F2)

Input: real per-action arm values, keeper labels, match labels from the corpus. Per replicate,
inject a keeper-level effect scaled so the induced between-keeper variance share equals a target
anchor; test detection against a **match-block permutation null** at α = 0.05; power = the detected
fraction.

Registered output: the **power curve at all three `ICC_ANCHORS`** — 0.015 / 0.020 / 0.026 — as the
shipped constant's docstring already promises. Per §6.1, if detection at the anchor is **< 0.8**,
floors/sampling are adjusted **first**; the ICC gate is not registered on an underpowered instrument.

### 5.4 ATT mode (row 5)

Input: real spells from `build_opportunities(frames, actions, home_team_id=..., model_metadata=...,
config=layer2_config(...))` — **Layer 2's design (D5), not `shot_arm_config`**. This is the review's
HIGH finding and it is worth restating as a rule rather than a fix: *a power curve is a function of
a design*, so `N_min` must be powered on the design it gates. Powering on `shot_arm_config` would
have anchored the relative effect on the **goal** rate (that config's `Y` is
`outcome_result_ids=(success,)`) while row 5 gates an **attempt** outcome — and that config's `Z`
(a shot attempt) is approximately Layer 2's `Y`. It is F3's category error recurring one level down,
inside the fix for F3.

Per replicate: subsample **spells at a target size, resampling whole match clusters** (never individual
spells — cluster-preserving, the same reason the null is match-blocked), inject a treatment effect at
the registered relative anchor, then `fit_propensity` → `propensity_match` → `estimate_att`; count a
detection when **`|ATT|/SE ≥ 2`** — the spec's own `live` definition, reused verbatim so the power
target and the gate agree.

**Matched-n is an OUTPUT, not a dial.** The loop cannot "subsample to a matched-n": matching consumes
spells and *yields* a matched count. Each replicate therefore records the pair
`(matched_n, detected)`, power is computed within matched-n bins, and **`N_min` = the smallest
matched-n bin at which power ≥ 0.80**. Bin occupancy is reported, and a bin below a minimum replicate
count is reported as unresolved rather than silently interpolated across.

**Anchors mirror the `ICC_ANCHORS` precedent** — a range, not a point, for the same reason the ICC
anchor is one: `ATT_RELATIVE_ANCHORS = (0.10, 0.15, 0.20)`, i.e. a 10/15/20 % relative change in the
outcome's base rate. The curve is reported at all three; **`N_min` is registered at 0.15**. The base
rate here is the **attempt** rate per final-third spell under `layer2_config` — measured and reported
alongside, so the relative anchor is interpretable in absolute terms and so the D2 decision is
auditable against the quantity it actually anchors on.

**Both outcomes, and the larger `N_min` wins.** Power is computed for `Y_attempt` **and**
`Y_close_attempt`; the registered `N_min` is the **maximum** of the two. This is the whole reason D3
pulls `outcome_max_distance_m` forward: `Y_close_attempt` has the lower base rate, so an `N_min`
derived on `Y_attempt` alone would be anti-conservative for precisely the outcome row 7 fires on.

### 5.5 The §3.3 shot-arm causal run (D4)

`scripts/validate_xshot_causal.py` ships complete and has never been run. It is run in this cycle
and its artifact written to **`docs/research/tf19_causal/xshot/`** — the path
`validate_xs_probe.py:290`'s `--entanglement` help already names as the banked result's home. (The
driver takes `--out`, so nothing was broken; the two just have to agree, and the shipped help string
is the one with a claim to defend.)

**Cost correction, against an earlier draft of this spec.** It consumes
`build_opportunities(..., config=shot_arm_config(model_metadata))` — a **different design** from
§5.4's `layer2_config`, so this is a **second** frames+actions corpus pass, not a free ride on
Layer 2's table. Cheaper than the ICC leg, but not free; D4's rationale is the record correction and
the unrun shipped driver, not shared data.

**No code change is proposed to the driver, and its result is not steered.** Three outcomes are all
valid and all get reported as-is:

| Result | Meaning for the record |
|---|---|
| `clears` | The xS arm's re-gate becomes **`joins`** (measured) — stronger than what 4.60.0 recorded |
| `inside_band` | 4.60.0's `joins_with_caveat` is CONFIRMED, and becomes measured rather than defaulted |
| `degenerate` (refusal) | Below `SHOT_ARM_MIN_CONTROL_CONVERSIONS` the driver refuses by design; the entanglement input stays honestly unmeasured and the caveat is re-described as such — a reportable result, not a failure |

**What this does NOT do: it does not rewrite `docs/research/tf19_pr3b_xs_v2/`.** That artifact's
citeability rests on its lock-commit blindness claim (`lock_commit == run_commit == 78ffc70`), and
retro-fitting a later input into it would destroy exactly the property that makes it evidence. The
measured entanglement is published alongside, and the re-gate consequence is stated in prose in both
artifacts. `regate_verdict` itself is a pure function — anyone can recompute the row from the two
published inputs, which is the point of having registered it.

---

## 6. Deriving the Layer 3 headroom threshold

The duty: *"**state `openGoal`'s units and observed range first** (a reader cannot currently tell
whether 0.02 is generous or unreachable), then set the threshold as a stated fraction of that range."*

Units are settled by F5 — a dimensionless open fraction, constructively in [0, 1]. The observed
marginal distribution of `openGoal` is measured on the corpus and reported (min/median/max +
quantiles), and the threshold on `median |Δ openGoal_with_GK|` is registered as a **stated fraction
of that observed range**.

**The fraction is committed BEFORE the measurement, and that ordering is load-bearing.** Measuring
the range first and choosing the fraction afterwards would make the threshold tunable to any desired
Layer 3 outcome — defeating the entire point of the derivation duty. Registered now, blind:
`LAYER3_HEADROOM_RANGE_FRACTION = 0.02` (2 % of the observed range). This is also the choice that
recovers the spec's own bare placeholder: `openGoal` is constructively bounded by [0, 1], so on a
corpus spanning most of that interval 2 % of the observed range lands at ≈ 0.02 — the number §6.4
guessed. The duty is discharged by making that number *interpretable and derived* rather than bare,
not by moving it.

**A boundary that must not be crossed, stated explicitly.** This derivation measures the **marginal
distribution of the shipped `openGoal` feature**. It does **not** run the ghost substitution and does
**not** compute Δ. Layer 3's probe — *"append the GK to the xS defender array, recompute
`openGoal_with_GK`... and measure how far the ghost substitution moves those feature values"* — stays
in PR-3b. The distinction is what keeps "derive the constant" from silently becoming "run the
experiment", and it is testable: the derivation script never imports `gkdv`.

**The landmark alternative was considered and rejected.** Because `openGoal ∈ [0, 1]` by
construction, a landmark-style threshold (a fraction of the *constructive* range) was available, and
round 2 withdrew the `D` quantile in favour of exactly that kind of landmark. It is not adopted here
for two reasons: the duty names the observed range explicitly, and — the substantive one — `D` sat on
**the row that decides H2**, whereas Layer 3 is **remedy routing**, not hypothesis adjudication. A
marginal range is treatment-blind and outcome-contrast-blind, which the spec itself conceded "was not
p-hacking"; the objection to `D` was about which row the constant sat on, and that objection does not
transfer.

---

## 7. Splitting verdict from routing (F4)

The conflation **is** the bug: `regate_verdict` answers "what did the probe say", while the routing
answers "what should we do about it", and only the second can legitimately depend on Layer 2.

- **`regate_verdict` stays byte-identical.** Every recorded verdict value stands — including
  `gated_clean_fail` and the xS arm's `joins_with_caveat` in shipped `metrics.json` artifacts. Pinned
  by a golden test over **all** valid input combinations.
- **New pure `regate_routing(verdict) -> str`** returns a routing token over a closed vocabulary
  (the `DAS_SOURCE_VALUES` pattern). `gated_clean_fail` maps to **`pending_layer2`**, not
  `gk_feature_engineering`.
- **H2 remains reachable ONLY through row 7** of `gkdv_discrimination_verdict` (PR-3b). This fix
  opens the channel; it does not pre-empt the decider.
- ADR-037 §4's routing rule is amended to match, and `docs/research/tf19_pr2/decision_table.md` gains
  a dated note that its routing sentence was superseded — the recorded **verdict** is untouched.

---

## 8. What the §6.4 amendment says

1. **The divergence.** xS `pass`/`joins_with_caveat` vs xCross `gated_clean_fail`. §6.4's premise —
   that the attempt arm reads flat — no longer holds for both arms, and per-arm treatment replaces
   any cross-arm assumption.
2. **A registered symmetry, and it is the important line.** §6.4 already holds that a **flat** probe
   is inadmissible as evidence about the world, because the feature contract predicts flatness. The
   converse is now needed and is the same argument run forwards: a **live** probe is evidence that
   *the model responds to keeper position* — Layer 1 responsiveness — and is **not** support for H1.
   Both H1 and H2 remain reachable only through Layer 2's model-free test. Without this, the xS
   `pass` invites exactly the mis-reading §6.4 exists to prevent.
3. **Row 5 is re-specified** per F3: `N_min` is an ATT constant with its own relative anchor; the
   ICC anchors stay with §6.1 where they are coherent. The `"at ICC 0.015–0.026"` clause is struck.
4. **The derived constants are registered**, with their measured provenance recorded inline.
5. **The routing amendment** (§7) is recorded against its pre-registered disclosure.
6. **The F6 record correction**, which reaches beyond §6.4 and must be made everywhere the claim
   appears: TODO.md's TF-19 row and the GKDV research-program note, CLAUDE.md's TF-19 bullet, and
   ADR-037 all describe 4.60.0's caveat as resting on "the banked SHOT causal arm". Corrected to
   name what it actually was — a registered default that became decision-relevant only when the
   probe passed — and then superseded by §5.5's measurement. The same wording appears in the CLI
   itself: `validate_xs_probe.py:290`'s help calls it the *"**banked** shot-arm causal result"*, so
   the help string is part of the correction, not just the prose. Stating the mechanism matters more
   than the value: a defaulted parameter documented as inert is a failure mode that will recur
   wherever a pre-registered function takes an input the expected verdict never reads.

---

## 9. Pre-registration integrity

**The legitimizing fact must lead in the ADR**, or this reads as post-hoc gate-moving: the amendment
was **itself pre-registered** — §6.4's Registration disclosures say *"`regate_verdict`'s routing needs
amending (ADR-037): `gated_clean_fail` must stop routing unconditionally to GK feature engineering,
since that hard-codes H1."*

- **`TF19_PROBE_ABS_FLOOR` is NOT touched** (the sibling registered disclosure). 4.51.0's
  `gated_clean_fail` stands as the shipping gate.
- **Every constant derived here is blind.** ICC and ATT power use *injected known* effects; the base
  rate and the `openGoal` range are *marginal* quantities. No derivation inspects a treatment
  contrast, and none can be tuned toward a preferred verdict.
- **No already-recorded verdict changes.**
- The resolved xgboost version is pinned into any artifact this cycle writes (§6.4's disclosure; the
  2.x/3.x `base_score` divergence makes an unpinned number non-citeable).

---

## 10. Testing

- **Both power modes get known-truth gates from BOTH sides** — power → ~1.0 at a large injected
  effect, and → ~α at zero injected effect. A one-sided "power is high" assertion passes identically
  when the simulator silently produces nothing. Each carries a **non-vacuity** assertion that the
  injection measurably moved the data, per the repo rule.
- **Clustering is asserted, not assumed:** a test that an i.i.d. shuffle of the block labels changes
  the reported power — otherwise "plasmode, not i.i.d." is a claim with no teeth.
- **`regate_verdict` golden** over every `(arm, probe_verdict, entanglement)` combination; a
  parametrized `regate_routing` test over the closed vocabulary; a test asserting
  `gated_clean_fail → pending_layer2`.
- **Builder legacy identity (D3 + D5):** with the covariate-treatment axis unset and
  `outcome_max_distance_m=None`, `config=None`, `xcross_config(model_metadata)` and
  `shot_arm_config(model_metadata)` all stay byte-identical — the existing regression check plus an
  explicit per-config identity test, since D5 changes treatment *semantics* and a default alone is
  not evidence.
- **The FIREWALL is a test, and the obvious version of that test cannot fail.** A call-count spy on
  `estimate_att` is **vacuous here**: the harness always calls it, so the spy cannot tell an injected
  outcome vector from an observed one and would pass identically under the defect it exists to catch.
  Required instead: the injected outcome vector carries a **provenance stamp**, the harness refuses
  any outcome vector lacking it, and the test asserts on the stamp — plus a **mutate → RED
  demonstration** (temporarily feed the observed `Y`, watch the guard fire) so "enforced by a test,
  not by discipline" is a demonstrated property rather than a claim. This is the same failure mode
  the repo has four documented instances of, and it appeared here in the gate the whole package rests
  on; it is recorded rather than quietly fixed. **Scope:** the guard binds the Layer 2 power harness
  only — §5.5's `validate_xshot_causal.py` computes a real observed-outcome ATT by design, and the
  test must not trip on it.
- **Covariate treatment, both sides:** spells straddling the 16.5 m threshold assign to the expected
  arm, AND a wide-spell case pinning that depth is `GK_r · cos(GK_theta)` — a test that would pass
  if `GK_r` were thresholded directly is vacuous for the bug it must catch.
- **`Y_far_attempt` partition arithmetic:** `ATT(close) + ATT(far) ≈ ATT(attempt)` on a fixture with
  multi-attempt spells — the exact case §6.4's N4 says makes the partition load-bearing. **State the
  reason inline: additivity is exact only because matching is Y-INDEPENDENT** —
  `propensity_match(ps, Z, *, target)` takes no `Y`, so all three outcomes share the same matched
  pairs and ATT is linear in `Y`. A future reader "fixing" the test by re-matching per outcome would
  break it for a reason the assertion alone would not explain. Two consequences: the three outcomes
  must use **identical row masks** (a spell with a missing `Y_close` but a valid `Y_attempt` breaks
  additivity for an entirely benign reason), and the comparison is `pytest.approx`, never `==`.
- **Fast CI smoke** for both power modes (§6.1 mandates it); the real curves are DGX runs.
- `causal/power.py` satisfies the public-module Examples + doctest gates.

---

## 11. The run & deliverable

DGX, from a **lock commit** recorded in the artifacts (the blindness idiom the xS-v2 run used:
`metrics.json` cites lock == run commit). Corpus: the 64-match GradientSports corpus the xS-v2 run
used, for continuity. Deliverables: `docs/research/tf19_s64_signoff/` — the ICC power curve at three
anchors, the ATT power curves at three relative anchors for both outcomes, the measured base rates,
the `openGoal` distribution, and the resulting registered constants — plus
`docs/research/tf19_causal/xshot/`, the §5.5 shot-arm entanglement artifact (D4), written by the
shipped driver in its own format.

**MEASURED on one WC2022 GS match (2026-07-26, `853432a`), replacing the estimate below.**
175,969 frames in → **2,224 scored** (drops: ball-far 66,544 / no-possession 51,758 /
ball-row-missing 46,547 / stride 8,896; conservation exact). **~40–57 min wall clock** per match
(the spread is download caching), **~5 GB RSS**, single-threaded at ~99 % of one core. Arm values
are two-sided and non-degenerate: mean −0.055 m², sd 0.88, range −15.1…+5.8, 1246 negative / 978
positive, **100 % non-zero** (so the identity-keyed-cache collapse-to-zero trap is absent).
Serial extrapolation to 64 matches is ~61 h; the box has 20 cores and the per-match work is
embarrassingly parallel with shards making concurrent writes safe, so the corpus pass should be
run parallel (~6–8 h), NOT serially. Three producer defects were found across four smoke runs —
missing possession derivation, double-keeper attribution, and mask misalignment — of which only
the second would not have crashed: it would have produced a full, plausible table whose ICC read
near-null. That is the run this smoke discipline exists to prevent.

**Cost, as originally estimated (superseded by the measurement above).** The three legs are not equally cheap. The
`openGoal` distribution is a pure-geometry pass over the xS extractor (cheap). The ATT mode *estimates*
on a spell table, which is small once built (for scale: the CROSS arm's banked table — still the only
causal table that exists — holds 23 966 opportunities / 669 treated). **Building** it is not free
though: neither Layer 2's nor the shot arm's spell table exists, so this cycle runs **two** distinct
frames+actions corpus passes (§5.4's `layer2_config`, §5.5's `shot_arm_config`), and Layer 2's
additionally joins TF-14 defensive-line and `bekkers_pi` pressure onto every spell. The **ICC mode is the
expensive leg**: a plasmode over *real* arm values requires the gkdv arms to be evaluated on the
corpus first — `delta_das` and `delta_threat_suppression`, i.e. accessible-space plus Spearman pitch
control on every domain frame — and neither arm may use a `PitchControlCache` (ADR-043: the cache
keys on frame identity, so a ghost frame would be served its twin's surface and every delta would
collapse to zero). Arm values are therefore computed **once** and persisted as the plasmode's input,
rather than recomputed per replicate; the replicate loop resamples that table. If the arm-value pass
proves to be the binding cost, it is a scoping conversation with the owner, not a silent reduction of
the corpus.

**If §6.1's ICC power is < 0.8 at the anchor**, that is a *finding*, not a failure to be worked
around: per §6.1, floors/sampling are adjusted first, and the ICC gate is not registered until it
passes. That outcome must be reported as prominently as a pass.

---

## 12. Out of scope (explicitly PR-3b, after sign-off)

Layers 0–3 **as experiments**; `gkdv_discrimination_verdict` and its precedence tests; the Layer 2
causal study itself; the owner validation run and PR-4. This cycle registers the constants and builds
the instruments those layers consume — it does not run them.

**The boundary in one line: Layer 2's DESIGN lands here; Layer 2's STUDY does not.** After D5 that
distinction carries real weight, so it is stated operationally rather than as a slogan. Landing here:
the treatment axis, `layer2_config()`, both outcome labellers, the confounder join, and the power
curve computed on **injected** effects. NOT landing here: **Layer 2's** observed-outcome ATT, its
positive/negative controls, its overlap and SMD diagnostics, and any Layer 2 verdict. §5.1's FIREWALL
is what makes the line enforceable rather than aspirational — the harness cannot emit Layer 2's real
contrast, and a test says so.

**The qualifier "Layer 2's" is load-bearing, not throat-clearing.** §5.5 computes a real
observed-outcome ATT — that is exactly what D4's entanglement study is — so a blanket "no
observed-outcome ATT lands here" would contradict §5.5 outright and stall any reader reconciling the
two. The firewall binds Layer 2's harness; `validate_xshot_causal.py` is untouched by it.

**§3.3 and Layer 2 are different causal studies, and the distinction is why one is in scope and the
other is not.** §3.3 (D4, in scope) measures **GK-confounder entanglement** — supportive context,
explicitly *"NOT a causal deterrence estimate"* per its own driver docstring — and feeds
`regate_verdict` as a single categorical input. Layer 2 (out of scope) is the **H1-vs-H2 decider**:
a different treatment (keeper depth binarised at the 16.5 m penalty-area line), different outcomes
(`Y_attempt` / `Y_close_attempt`), a different confounder set, and its own positive/negative controls
and overlap diagnostics. They share the `causal/` estimators and the opportunity builder; they share
neither an estimand nor a verdict. Running §3.3 does not run, prejudge, or partially answer Layer 2.

---

## 13. Attribution / C4 / retrain

No new aggregator, no backend, no trained model → **C4-free (count stays 32)**; confirm by running
`/c4` at commit-prep rather than asserting it. In no default xfn list, no VAEP consumer → **no
retrain trigger**. Attribution: Abadie & Imbens (2006) matching SEs are already cited in NOTICE via
ADR-015; the plasmode-simulation idiom gets a NOTICE entry per the attribution discipline.
