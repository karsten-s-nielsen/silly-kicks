# TF-19 A+2 — physics-arm instrument validity + responsiveness + named-keeper validation — design

**Status:** DRAFT for review. Brainstormed 2026-08-28.
**Origin:** TF-19 GKDV closeout. The physics arms (`delta_das`, `delta_threat_suppression`) shipped
gate-independent (ADR-043; batched ADR-075); the §6.1/§6.3 validation harness constants,
`aggregate_by_keeper`, `EXPECTED_DIRECTION` and Layer-4 `behavioural_anchoring_verdict` shipped; the
§6.1 ICC / §6.3 ATT power sign-off already ran (4.68.0, `docs/research/tf19_signoff_power/`). What
remains of the TF-19 §6.4 stack is **Layers 0-3** + the composing `gkdv_discrimination_verdict`. This
cycle takes the **run-first, GPU-free slice** of that remainder plus the owner validation run.
**Version / PR-S / ADR numbers are assigned at commit-prep after `git fetch && git merge origin/main`
— this doc uses `ADR-082`.**

Parent spec: `docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md` (the §6.4 layer
definitions; this design implements Layers 0-1 + the named-keeper validation, physics arms only).

---

## 1. Problem

TF-19's physics arms are shipped and gate-independent, but they have **no instrument-validity or
responsiveness probe** — the two §6.4 layers that answer "does this arm actually respond to a keeper
moving, and is that response real rather than an artifact?" before any keeper-level number is
interpreted. Layer 4 (behavioural anchoring) shipped because it gates the already-shipped ICC; Layers
0-1 did not. And the owner-facing payoff — *does the metric rank known sweeper-keepers (Alisson,
Neuer) as strong deterrents?* — has never been produced, so the shipped arms have no demonstrated
face validity.

The attempt-arm channels (xS, xCross) already have their probes (`tracking/_model_eval.py`:
`evaluate_xs_probe`, `XS_PROBE_DOSE_LADDER`, `gk_substitution_probe`) and already ran
(`docs/research/tf19_pr3b*`); xCross is `gated_clean_fail`, xS is `joins_with_caveat`. They are out of
scope here (see §2, §3.1).

## 2. Goals / non-goals

**Goals**
- **Layer 0 (instrument validity):** a per-physics-arm dose-imposition probe with a registered
  saturating-dose positive-control rule, emitting a pure-function verdict.
- **Layer 1 (responsiveness):** a per-physics-arm Regime-I-vs-Regime-O responsiveness statistic
  reusing the shipped `gk_med ≥ RATIO × max(nd_med, placebo_p95)` idiom, extended with paired-vector
  controls.
- **A (named-keeper validation):** an owner-run expected-sign report over WC2022 GS keepers, reusing
  the shipped `aggregate_by_keeper` / `EXPECTED_DIRECTION` / `behavioural_anchoring_verdict`.
- **A caller-supplied keeper-identity resolver seam** so per-keeper aggregation is provider-agnostic
  and SB360 is not foreclosed.
- All of it **reported-not-gated**, with the Layer 0/1 verdict *vocabulary* shaped to be consumed by
  the future `gkdv_discrimination_verdict` decision table (rows 0, 2, 4) without redesign.

**Non-goals**
- The attempt-arm channels (xS/xCross) — their probes exist and already ran.
- **Layer 2** (the model-free causal ATT decider) — heaviest slice, and already measured
  underpowered on WC2022 (`N_MIN_MATCHED` unreachable, 151 treated spells; `tf19_signoff_power`); a
  run now returns `underpowered` = not evidence. Genuinely blocked on corpus expansion.
- **Layer 3** (feature-headroom remedy probe) — deferred to the next slice.
- The composing **`gkdv_discrimination_verdict`** — deferred; no gate flips this cycle.
- Any retrain, re-materialize, artifact change, or new C4 aggregator.
- GPU / trained-model dependency (the physics arms are model-free).

## 3. Settled decisions (with rationale)

1. **Physics arms only.** Layer 1 forbids cross-axis magnitude comparison (a m² delta and a
   probability delta are non-commensurable — a stated ADR non-goal), so each channel is judged on its
   own dimensionless ratio and bundling channels buys **zero** analytical value. The physics arms are
   the load-bearing gap (they *ship* per parent-spec row 7) and are GPU/model-free; the attempt arms
   already have their probes and are gated. Re-probing them re-enters a settled interpretation problem
   (Chesterton's Fence on shipped, gated code).
2. **Reported-not-gated.** The composing verdict is deferred, so nothing here flips a gate. The Layer
   0/1 verdicts ship as **registered pure functions** (the `regate_verdict` / `behavioural_anchoring_
   verdict` idiom) whose measured outputs are reported this cycle and consumed by the future table
   unchanged.
3. **Honest per-provider / per-arm degradation — `arm_unscoreable` is a first-class verdict, and the
   parent's S1 diagnostic is deliberately NOT ported (Chesterton's-Fence clearance).** The parent
   spec's `saturating_dose_unsupported` carve-out (Review Round 1 S1) guards a *specific* mode: the
   boosted-tree attempt models flat-extrapolate when the saturating dose lands in a thinly-covered
   region of their TRAINING hull, so a flat response there is an OOD artifact, not brokenness. **The
   physics arms are model-free** (deterministic DAS / pitch-control geometry) — at a direct
   keeper→goal-line teleport over the ball-near-goal domain there is **no trained-hull OOD-flatness
   confound**, so a flat response IS a true diagnosis. The parent's training-support diagnostic is
   therefore **deliberately not ported** — it guards a mode the physics arms cannot enter. What
   remains, and what `arm_unscoreable` names honestly, is a *different* concept — the arm is **not
   computable / not adequately supported on this input**: (a) velocity-less ΔDAS (provider capability;
   honest-NaN `DasUnscoreableError`, never a fabricated 0, ADR-043/054/063), or (b) a domain thinner
   than `MIN_DOMAIN_FRAMES` (§4.1). It is distinct from `instrument_void` (scoreable, but flat under
   the saturating dose). `arm_unscoreable` is thus the parent's outcome *by analogy of role*, not a
   rename that inherits its training-hull reasoning.
4. **Caller-supplied keeper-identity resolver — SB360 is not foreclosed (but the SB360 adapter is
   to-be-built, not present today).** SB360 freeze-frames carry no *intrinsic* recurring identity
   (`shape_snapshots` numbers rows), and today's freeze-frame path yields only a defending-GK *row* (a
   bool, via `is_goalkeeper` / `defending_gk_visible`), **not an identity**. The id is *recoverable*
   via a **to-be-built, driver-side adapter**: the StatsBomb roster parser already exists **scripts-side**
   (`scripts/_sb_raw.py::parse_roster` → `{player_id: {name, jersey, team, position}}`), so joining its
   `position == "Goalkeeper"` id (keyed by the ADR-062-resolved real `team_id`) onto the defending-GK
   snapshot row is a plan-time build, assembled in the driver (which may call `parse_roster` — there is
   no roster parsing library-side, ADR-062: "no player identity at all"). A+2 ships the resolver
   **seam + the tracking native-id adapter**; the SB360 roster-join adapter is a *documented contract*
   deferred with the SB360 run (other session). The seam shape is fixed here, so SB360 is not
   foreclosed. This applies the ADR-062 lesson: *"a provider port that declines to name an identity it
   can DERIVE from context it already receives is not conservative — it silently breaks consumers."*
5. **Tracking-primary run, SB360-open design.** The A+2 *run* is WC2022 GS (velocity + identity,
   both arms). SB360 stays reachable via the resolver seam for the Δthreat arm (zero-velocity pitch
   control) with a roster-injected id; ΔDAS remains NaN on SB360 (velocity). SB360 execution is the
   other session's; A+2 must not foreclose it.
6. **ADR-037 import allowlist preserved.** `gkdv/` imports `tracking._das` ONLY (via `_das_port.py`);
   `tracking/` never imports `gkdv/`. The probe consumes gkdv arms + the ghost engine, so it lives in
   `gkdv/`; any cross-package assembly lives in the `scripts/` driver.

## 4. Architecture

### §4.1 Dose imposer + Layer 0 (`gkdv/_probe.py`)
A per-arm dose-imposition primitive over the shipped ghost engine, restricted to the gkdv domain
(alive ball ∧ in-possession team attacking ∧ ball within `domain_ball_to_goal_m` of the attacked goal
∧ defending-GK row present; the ball carrier pinned once, as in the arms). Three doses on the SAME
frames:
- **realistic** — observed ghost − actual displacement, `|δ| ≥ 2 m` (the shipped ghost's ~1.1 m MAE
  makes 2 m the meaningful floor, inherited from the parent spec);
- **ladder** — fixed `{2, 3, 4} m` (mirrors `XS_PROBE_DOSE_LADDER`);
- **saturating** — keeper → goal-line centre, and keeper → goal-relative `x = 30 m` (positive
  control; both are realizable keeper positions).

Registered per-arm rule (pure function, `regate_verdict` idiom), stated **exactly as the parent spec
registers it — as the VOID condition** (so the pre-registration is copied, not re-derived):
`instrument_void` iff the saturating median |Δ| is **not** ≥ `SATURATING_MULTIPLE` × the arm's own
realistic median |Δ| **AND** **not** > the arm's own paired-vector placebo p95; otherwise
`instrument_valid`. Equivalently by De Morgan: `instrument_valid` iff `saturating ≥ 5× realistic`
**OR** `saturating > placebo p95` (a lenient bar — passing *either* test suffices, which is the
parent's intent). **Zero-baseline guard (as-built, 2026-08-29):** the multiple leg additionally requires
`real_med > 0` — `saturating ≥ 5 × 0` is trivially true, and ΔDAS is zero-dominated (so `real_med == 0`
is plausible on the one arm this pass scores), which would vacuously validate a DEAD instrument; when
there is no realistic baseline the placebo leg is the sole backstop. The pooled medians (`real_med`,
`sat_med`, `placebo_p95`, `gk_med`, `nd_med`) are RECORDED in the artifact so the verdict is auditable.
`SATURATING_MULTIPLE = 5` (registered in the parent spec as the literal `5×`, `:1061` — this cycle
gives it a *named code constant* holding that parent value; it is copied, not re-derived). **Two
short-circuits, evaluated FIRST, both → `arm_unscoreable`:** (a) an arm not scoreable at the imposed
dose on the provider (velocity-less ΔDAS); (b) a domain thinner than a registered `MIN_DOMAIN_FRAMES`
floor — the pooled realistic/saturating medians and placebo p95 are noisy on a thin domain and could
fire `instrument_void` on noise, so below the floor the verdict is `arm_unscoreable` (insufficient
support), the direct analogue of the keeper-count limitation `behavioural_anchoring_verdict` states.
`MIN_DOMAIN_FRAMES` is derived and recorded in the artifact (stated, not guessed). **It applies to the
POOLED-corpus domain, not a single shard** — the Layer-0/1 verdicts are pooled statistics (medians,
placebo p95) computed in a **reduce over all shards' per-frame values** (§4.5), never per match; a
single match's domain is thin by construction, so a per-shard floor would fire `arm_unscoreable`
everywhere. Verdict set: `{instrument_valid, instrument_void, arm_unscoreable}`.

### §4.2 Layer 1 (responsiveness) + paired-vector controls (`gkdv/_probe.py`)
Two registered regimes, assigned in advance so they cannot be swapped post hoc: **Regime I** (imposed
dose — discriminating) and **Regime O** (observed ghost — the shipped metric). Per-arm dimensionless
statistic, reusing the shipped idiom verbatim: `gk_med ≥ RATIO × max(nd_med, placebo_p95)`. **`RATIO`
is a NEW registered physics-arm constant `PHYSICS_ARM_PROBE_RATIO = 2.0`, not an inheritance** — the
honest reading of the code is that *both* existing constants are model-specific (`TF19_PROBE_RATIO`'s
own comment reads "Pre-registered TF-19 **xCross** viability threshold", `_model_eval.py:26`;
`XS_PROBE_RATIO` is the xS model's), and the physics arms are neither xS nor xCross. The genuine
authority is the parent Layer-1 registration (`:1085`), which fixes the *form* of the idiom but
specifies **no numeric value**. So we register 2.0 as a deliberate new choice — the magnitude both
attempt probes happen to use — rather than dressing it as an inheritance via a constant's prefix
(a "quote the meaning, not the name" case). **No `abs_floor`** — Layer 1 is "comparable but NOT
decisive"; the parent Layer-1 registration (`:1085`) and the xs rule both omit it (the `abs_floor`
lives only on the xcross gate, verified). Extended to the physics arms with **paired-vector controls**.

> **CORRECTED (as-built, 2026-08-29).** The original wording below — "the nearest defender **plus** `R`
> random outfielders displaced by the identical vector (**this control set is also the placebo band**)"
> — is a design flaw: one COMBINED control moving `R+1` players makes `nd_med` and `placebo_p95` the
> SAME array, so the idiom's `max(nd_med, placebo_p95)` is decorative, and it compares a 1-player keeper
> move against an `R+1`-player control (asymmetric, biased toward `not_responsive`). The parent idiom
> (`_model_eval.py:405-439`) moves **ONE player per control**: the nearest defender ALONE (`nd`) plus a
> band of **single**-outfielder placebo replicates. As-built, `paired_vector_controls` returns
> `{"nearest": ..., "placebo_0": ..., ..., "placebo_{R-1}": ...}` — each a frame with ONE defending
> outfielder displaced by the per-frame vector — so `nd_med` (nearest, one player) and `placebo_p95`
> (band of single players) are DISTINCT and the `max` is meaningful.

The nearest defender (`nd`) and `R` single-outfielder placebo replicates are each displaced by the
*identical per-frame vector*; the placebo pool is the defending team's outfielders (a keeper-positioning
control relocates a DEFENDER, not an attacker). `R` registered (default 3, matching the probe idiom).
**No cross-axis magnitude comparison — ever** (stated ADR non-goal). Verdict per arm per regime:
`{responsive, not_responsive, arm_unscoreable}`.

### §4.3 Caller-supplied keeper-identity resolver (`gkdv/`)

> **SUPERSEDED (as-built, 2026-08-29 — see ADR).** This gkdv-local `resolve_defending_keeper_id`
> resolver was **DROPPED**. The SB360 first-class-provider cycle shipped ONE resolver
> `tracking.resolve_keeper_identities` (ADR-078/ADR-055 single-source); a second gkdv-local identity
> path would violate ADR-055, and ADR-037 confines gkdv to `tracking._das`. The driver (§4.4/§4.5)
> consumes `tracking.resolve_keeper_identities` DRIVER-side (`identity="native"` for velocity-bearing
> GS; the roster path for SB360). Only the arm-direction-key seam (§4.4) was built as written here.
`resolve_defending_keeper_id(frames, *, roster=None, on_unresolved="drop")` → a per-scored-frame
defending-GK id column feeding `aggregate_by_keeper` unchanged:
- **tracking:** the native frame GK `player_id` (identity-bearing providers);
- **SB360:** roster-resolved — the defending `is_goalkeeper` row + resolved `team_id` → the roster's
  `position == "Goalkeeper"` player id/name for that team;
- **unresolvable** (no roster, ≠1 GK for the team, missing team) → **dropped and counted**, never a
  fabricated id.
Pure; never mutates `frames`. **Dependency inversion:** the resolver consumes an **injected `roster`
dict** and does not parse anything itself; the roster parser is scripts-side
(`scripts/_sb_raw.py::parse_roster`, consistent with §3.4 — there is no roster parsing library-side),
and the `scripts/` driver calls it and injects the dict. The seam shape is fixed here; the SB360
roster-join adapter is deferred with the SB360 run.

### §4.4 The "A" named-keeper validation (driver + `gkdv/`)
Pipeline: arm values on the domain → `resolve_defending_keeper_id` → `aggregate_by_keeper`
(`min_nonzero`, `min_games ≥ 2`) → per-arm `EXPECTED_DIRECTION` (negative = deterrent) + Layer-4
`behavioural_anchoring_verdict` (shipped) + a **named-keeper expected-sign report**.

**Arm-key normalization (a required seam, not free reuse).** The arm columns are `delta_das`
(`_arms.py:281`) and `delta_threat_suppression` (`:341`), but `EXPECTED_DIRECTION` is keyed
`{delta_das, delta_threat}` (`_validate.py:76`) — so the threat arm's column name does **not** match
its direction key, and this §4.4 driver is `EXPECTED_DIRECTION`'s **first consumer** (today it is
referenced only by its own definition + test), so there is no existing normalization to inherit. The
driver pins one canonical mapping `_ARM_DIRECTION_KEY = {"delta_das": "delta_das",
"delta_threat_suppression": "delta_threat"}` at the boundary — leaving the physics function names
intact — and a test asserts **every** arm column resolves to an `EXPECTED_DIRECTION` entry (so a
future arm cannot silently drop its sign check). This is a real rename the "reuse unchanged" ethos
otherwise hides.

**Reach must be measured on the BINDING census (both floors), per arm.** `gate_eligible` requires
`min_nonzero ≥ 20` **AND** `min_games ≥ 2` (`_metric.py:39-42`), and **ΔDAS is exactly 0 whenever the
displacement moves no accessible-space boundary** (`:35`), so for the DAS arm `min_nonzero` is the
*binding* floor, not `min_games`. The raw sign-off census (**41 keepers, 8 single-match**,
`_validate.py` docstring) bounds `min_games` alone → ≤ 33; it is an **upper bound**, not the eligible
set. So the plan/driver reports the **per-arm `gate_eligible` count** (ΔDAS and ΔThreat separately —
ΔDAS is the zero-heavy one) from the sign-off arm-values (or recomputes it) **before** the run, and
the named-keeper eye-test is caveated to whatever that measured per-arm count supports (if a tercile
falls below three keepers for an arm, its anchoring reads `uninterpretable` for a reason unrelated to
the metric, and the report says so). The "established, not assumed" claim attaches to the *measured
per-arm gate_eligible number*, never to the 41/8 raw census.

The named-keeper report: known sweeper-keepers (Alisson, Neuer) expected strongly negative vs
line-keepers, under the ≥2-match rule, with 0-minute / single-match keepers (Ter Stegen 0 min; Onana
descriptive-only) stated honestly. A **pre-registered expected-sign summary** (a named-keeper sign
table locked before the run) makes it a recorded eye-test, not a post-hoc read.

> **AS-BUILT (2026-08-29, corrected).** Two things are pre-registered, and they are DIFFERENT: (i) the
> arm-level EXPECTED DIRECTION (`negative` == deterrent, `EXPECTED_DIRECTION` + `_ARM_DIRECTION_KEY`),
> applied per keeper as `sign_matches_expected`; and (ii) THE NAMED-KEEPER PRIOR — `NAMED_KEEPER_PRIOR`
> (`{Alisson, Neuer} → negative`) with the §4.4-stated caveats (`{Ter Stegen: 0-min, Onana:
> descriptive-only}`), **LOCKED in code 2026-08-29 before the owner run** and stamped into `metrics.json`
> as the pre-registration record. Only the *observed* sign is unlockable; the *expected* prior for known
> keepers IS lockable, and locking it is what makes the Alisson/Neuer eye-test confirmatory rather than
> post-hoc. The driver emits the per-KEEPER sign table `named_keeper_signs.parquet` (`player_id`,
> counts, `mean`, `observed_sign`, `sign_matches_expected`, `gate_eligible`), and when an owner-injected
> `--keeper-names-json {player_id: name}` is supplied it joins `keeper_name` + runs the confirmatory
> `named_keeper_prior.check` (per named keeper: matched keepers, observed signs, `meets_prior` — TRUE
> only when the name resolves to ≥1 keeper AND every match meets the prior, so an unresolved name never
> falsely confirms). Name resolution is INJECTED (dependency-inverted), never parsed here. Anchoring runs
> on the gate-ELIGIBLE subset (the S6.1 floors thin the surface first).

### §4.5 Driver + artifact (`scripts/`)
One owner-run corpus driver: ADR-052 `for_each` (per-match shards, resume-safe, conservation +
injectivity), ADR-037 provenance (`require_clean_tree`, `run_commit`/`run_tree_dirty`, `--allow-dirty`
records `dirty:true`). **Two-stage by grain, and the split is load-bearing:** the sharded `for_each`
map emits **per-frame** arm/dose/control values per match (the shardable, resumable unit); the
**Layer-0/1 verdicts and keeper aggregation are computed in a REDUCE over the concatenated shards**,
NOT per shard — the verdicts are pooled-corpus statistics (medians, placebo p95, the
`MIN_DOMAIN_FRAMES` floor), and a single match's domain is thin, so a per-shard verdict would be
`arm_unscoreable` everywhere. The reduce → `docs/research/tf19_instrument_responsiveness/`,
recording every registered constant, the named-keeper
sign table, and the **provider-support matrix** (velocity + identity requirements; which arm is
scoreable on which provider). Enrolled in the ADR-056 artifact-driver population + ADR-052 population.

> **AS-BUILT (2026-08-29).** The authoritative artifact is `metrics.json` (+ `named_keeper_signs.parquet`)
> — named `metrics.json` so the ADR-056 staleness detector, which globs `metrics.json` keyed on
> `input_contract.driver`, actually sees it (the driver is enrolled in `_DECLARING`, and its
> `input_contract()` declares every registered threshold so a change moves the digest). `metrics.json`
> records: per-arm Layer-0/1 verdicts WITH the pooled medians they rest on; the `registered_constants`
> block; the `provider_support` matrix; the `keeper_identity` block (dropped-AND-counted resolution
> totals). **Parallel-safety:** a `--match-ids-json` worker writes shards + a per-worker manifest ONLY;
> the pooled reduce runs on a final UNPARTITIONED pass (a partition worker sees a race-dependent PARTIAL
> shard set, so it must not emit a corpus verdict).

## 5. Testing strategy

**Red-before-green (ordering, not just content).** Each verdict test lands as a *failing* assertion
first — write the `instrument_void`-on-flat-fixture / `arm_unscoreable`-on-velocity-less / arm-key-
resolves assertions, watch them fail, then implement. Non-vacuity (below) is the guard's *content*;
red-first is its *ordering* — both are required.

- **Pure-function unit tests, non-vacuous** (a mutation that *should* flip a verdict does):
  - Layer 0: a dead instrument (arm flat under the saturating dose) → `instrument_void`; a live one →
    `instrument_valid`; velocity-less ΔDAS → **`arm_unscoreable`**, asserted distinct from
    `instrument_void`; a domain below `MIN_DOMAIN_FRAMES` → **`arm_unscoreable`** (insufficient
    support), also asserted distinct from `instrument_void` (a thin domain must not read as "broken").
  - Layer 1: both sides of the ratio (a responsive arm passes, a flat one fails); paired-vector
    control applies the *identical* vector (assert the control displacement equals the keeper's).
- **Arm-key resolution (F1 guard):** a test asserts every arm column (`delta_das`,
  `delta_threat_suppression`) maps through `_ARM_DIRECTION_KEY` to a present `EXPECTED_DIRECTION` entry
  — so a future arm cannot silently skip its sign check (non-vacuous: an unmapped arm fails).
- **Identity resolver:** native id passes through; a synthetic SB360-shaped frame + roster →
  per-keeper aggregation resolves to the named GK; unresolvable (no roster / ≠1 GK) → dropped and
  counted, never fabricated.
- **Driver:** `for_each` conservation + injectivity; provenance wiring (`require_clean_tree` from
  `main()`, no bare `rev-parse`); an empty shard still writes.
- **e2e backstop:** the shipped ghost/arms/`aggregate_by_keeper`/`behavioural_anchoring_verdict`
  suites pass unchanged (this cycle is additive).
- GPU-free, model-free → everything runs in the normal `-m "not e2e"` suite.

## 6. Constraints

- ADR-037 import allowlist (gkdv → `tracking._das` only; no `tracking` → `gkdv`); probe in `gkdv/`,
  cross-package assembly in the driver.
- Reported-not-gated; no gate flip (`TF19_PROBE_ABS_FLOOR` untouched; `regate_verdict` routing
  untouched); no retrain, no re-materialize, C4-free.
- All registered constants locked in code before the owner run (pre-registration). **Copied from the
  parent spec** (its literal value, given a named code constant): `SATURATING_MULTIPLE = 5`, the 2 m
  realistic floor, the {2,3,4} ladder. **New physics-arm registrations** (the parent Layer-1 idiom
  fixes the form, not the value — so these are honest new choices, derivation recorded in the
  artifact): `PHYSICS_ARM_PROBE_RATIO = 2.0` (the magnitude both attempt probes use; NOT inherited
  from the model-specific `TF19_PROBE_RATIO`/`XS_PROBE_RATIO`), `MIN_DOMAIN_FRAMES` (the pooled-corpus
  Layer-0 support floor), `R` (paired-vector control count, default 3), the two saturating positions
  (goal-line centre; goal-relative x = 30 m). No `abs_floor` (Layer 1 is comparable-not-decisive).
- Lint at CI scope; bare pyright; full `-m "not e2e"`.
- Single feature branch, single commit, single PR. No version numbers until commit-prep. No commit
  without explicit owner approval.

## 7. Execution ordering (review-tractable; NOT commit boundaries)

1. `gkdv/_probe.py` dose imposer + Layer-0 verdict (+ `arm_unscoreable` short-circuit) + unit tests.
2. Layer-1 responsiveness statistic + paired-vector controls + unit tests.
3. `resolve_defending_keeper_id` seam + unit tests (native + SB360-roster + unresolvable).
4. The `scripts/` owner-run driver — two-stage: ADR-052 `for_each` map (per-match shards of per-frame
   values) + a **reduce** computing the pooled Layer-0/1 verdicts + the per-arm `gate_eligible` reach
   census + keeper aggregation; ADR-037 provenance; enroll in the artifact-driver / corpus-driver
   populations. Driver tests incl. the reduce-not-per-shard verdict.
5. Docstrings, ADR (`ADR-082`), CLAUDE.md gkdv-bullet extension, NOTICE unchanged, C4 verify (no new
   aggregator). Full CI-faithful gate + /final-review + /c4.
6. Owner run on WC2022 GS → `docs/research/tf19_instrument_responsiveness/` (the artifact is
   owner-run; the driver + harness are what this cycle ships).

## 8. Known limits (stated, not discovered)

- **ΔDAS is unscoreable on velocity-less providers** (SB360) by construction; the Δthreat arm and the
  ghost engine work there, so SB360 gets a threat-arm-only named-keeper validation *if* the other
  session runs it with a roster-injected id.
- **Mid-match GK substitution** makes "the team's GK" non-unique for a full match; the roster resolver
  maps the starter and is honest about subs (resolvable via substitution events if needed). For the
  full-match eye-test keepers this is a no-op.
- **A named-keeper eye-test is descriptive, not a gate.** It demonstrates face validity; the causal
  discrimination (H1 vs H2) is Layer 2, out of scope and underpowered on the current corpus.
- **Layer 2 remains blocked on corpus expansion** — nothing here changes that; A+2 deliberately does
  not touch it.

## 9. Data-support matrix

| Provider | velocity | recurring identity | ΔDAS arm | Δthreat arm | named-keeper A |
|---|---|---|---|---|---|
| GS / Sportec / SkillCorner / Metrica (tracking) | yes | yes (native) | ✅ | ✅ | ✅ (native id) |
| SB360 (freeze-frame) | no | no (roster-injectable) | ❌ `arm_unscoreable` | ✅ (zero-velocity) | ✅ threat-only, roster id |

`resolve_defending_keeper_id` + the `arm_unscoreable` verdict are exactly the two seams that make this
matrix honest rather than fabricated.
