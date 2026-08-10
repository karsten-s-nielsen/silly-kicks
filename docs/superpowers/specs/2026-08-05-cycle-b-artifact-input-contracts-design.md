# Cycle B — research-artifact input contracts and registry completeness — design

**Status:** approved 2026-08-05. **The ADR number and version are assigned at COMMIT-PREP, read off `main` at that moment -- never pre-claimed.** A hard `the Cycle B ADR` in this document and in 23 source files was taken by a concurrent session mid-cycle (4.76.0 / the Cycle B ADR, SB360); 4.77.0 and 4.77.1 went the same way. Code and tests refer to this work as **Cycle B**, a slug no other session can claim. Successor to
`2026-08-03-adr051-closeout-and-artifact-validity-design.md`, which specified this work inline as
§5.1–§5.8 while PR 5 was in flight. That material is consolidated here; the PR 5 spec remains the
record of how these items were found.

**Goal.** Stop "is this recorded number still valid?" being archaeology. Every artifact declares what
its numbers depend on, every registry derives its own surface, and the answer to "what does this PR
invalidate?" becomes a list a machine produces rather than a call graph a human walks.

**Why now.** PR 5 changed a geometry transform and three research artifacts silently went stale. The
only thing that found them was someone tracing `causal/opportunities.py` by hand — after seven
review rounds had missed it. The mechanism below is what that tracing should have been.

## Scope

**Seven items, one PR, three phases.**

| item | what |
|---|---|
| 9 | research-artifact input contracts |
| 10 | `ARTIFACT_DRIVERS` completeness gate |
| 23 | repair the GS input-convention guard |
| 24 | pin the C4 aggregator count |
| K2 | the ADR-020 dup-`action_id` gate: no atomic coverage, tautological meta-assertion |
| K9 | the provenance gate checks a driver's SOURCE, never its OUTPUT |
| 26 | ADR-003's NaN-safety registry, and three `*_xfns` leakage guards, are pinned by FLOORS with no two-directional tie to the public surface |

Six of the seven are one genus: **a registry, count, or contract a human maintains beside a mechanism
that could derive it.** Item 23 is here because it must land before PR 6 and is small.

**What the cycle actually delivers, stated narrowly enough to match the design:** *derive the
POPULATION mechanically, justify the EXCLUSIONS humanly.* §3.1 recommends keeping a reasoned exemption
registry rather than a smarter matcher, so "derive, don't maintain" would overstate it. The judgement
about what belongs stays human and reviewable; what stops being human is the enumeration.

**Deferred to Cycle C** (registered so they stop drifting between documents): item 25 (`from_hub`
revision pinning + publication as a pipeline step), item 14 (GS corpus taxonomy re-registration),
item 15 (the fold-count ship rule with no magnitude floor), item 22 (gate the taxonomy decision at
the next `train_gk_completion` run), K1 (the `_a0/_a1/_a2` slot columns undocumented in
`feature_glossary`), K7 (`validate_xcross_causal` re-extracts on aarch64 against x86-stamped
weights).

**Not in either cycle:** the TF items in `TODO.md`'s On Deck, PR 6's aggregators, PR 7's ghost
weights.

### Disposition of every K-registry entry

An earlier revision presented a two-bucket partition (in / deferred) over a registry that also holds
resolved, accepted-design and subsumed entries -- so three buckets existed implicitly and none was
written down, and five of ten entries were named nowhere. The partition is the cycle's own thesis
applied to itself: derive the population, justify the exclusions.

| K | disposition |
|---|---|
| K1 | **deferred** to Cycle C -- glossary `_a0/_a1/_a2` coverage |
| K2 | **in scope**, §3.3 |
| K3 | **resolved in 4.74.0** -- `tf19_pr3b` was rewritten with provenance; independently re-verified here (7 of 7 `docs/research/**/metrics.json` carry a `run_commit` and none is dirty) |
| K4 | **not a defect** -- `git_provenance` treating git-unavailable as dirty is the correct fail-closed reading; recorded as an accepted design note |
| K5 | **deferred** to Cycle C, with item 25 -- the one-variant-per-repo claim is adjacent to `from_hub` revision pinning but not identical to it |
| K6 | **superseded** by item 23, which is in scope |
| K7 | **deferred** to Cycle C -- aarch64 re-extraction axis |
| K8 | **subsumed by item 10** -- it reports the `ARTIFACT_DRIVERS` floor weakening as the registry grows (now `>= 6` against 18), which is exactly what §3.1 replaces. Stated rather than inferred so the entry can be burned down. |
| K9 | **in scope**, §2 |
| K10 | **not a defect** -- the placebo-band quantization is a resolution limit, not an error; it stays a live caveat on cited numbers and needs no code change |

---

## 1. Mechanism — declared input contracts (item 9)

### 1.1 The declaration references SYMBOLS, never values

```python
# scripts/validate_xshot_causal.py
def input_contract() -> dict:
    return declare_inputs(
        covariates={"shot_arm": SHOT_ARM_CONFOUNDERS,
                    "gk_block": shot_arm_config({}).gk_block},
        geometry_version=_geo.GEOMETRY_VERSION,
        extractors=("silly_kicks.tracking._xshot_occurrence",),
    )
```

A driver declares **which symbols** its numbers depend on, not what those symbols currently contain.
When `SHOT_ARM_CONFOUNDERS` gains a column or `GEOMETRY_VERSION` bumps, the digest moves without
anyone editing the driver.

This is the difference between the design and "a human writes a list". The residual failure mode is
"forgot to reference a symbol at all" -- narrow and visible -- rather than "typed a list that later
went stale", which is what a literal declaration would ship.

`declare_inputs` returns `{version, covariates, geometry_version, extractors, digest}`, written into
the driver's `metrics.json` beside `run_commit`. Deliberately the same shape as ADR-050's
`feature_contract`, because that pattern is already built, reviewed and trusted here.

### 1.2 CI re-derives and compares

A gate imports each registered driver, calls `input_contract()`, digests it against live code, and
compares to every committed artifact that driver produced. A mismatch means the artifact's inputs
moved since it ran.

**Warn, do not raise.** An artifact is not a serving path, so a mismatch is not a load failure. It
must surface at PR time rather than at read time -- that is what converts PR 5's hand-tracing into a
list.

### 1.3 Invalidation is computed, never maintained

The "what does this PR invalidate?" report is derived by comparing declarations against the diff.
There is no stored list of artifact-to-input edges to fall out of date. A central reverse index was
considered and rejected for exactly that reason -- it rots the moment a driver is added without
touching it, which is how `ARTIFACT_DRIVERS` reached a floor of 6 against 18 entries.

### 1.4 Known limit, declared rather than discovered

**This catches code drift, not under-declaration.** A driver that never references `theta` digests
stably forever. Two alternatives were considered:

* **Derive the declaration automatically** from imports and config objects. Rejected: PR 5 showed the
  dependency runs through `causal/opportunities.py` into extractors two hops away, so a shallow
  derivation re-creates the caller-sweep blind spot *in code*, and a deep one is a static-analysis
  project. It is the trap option -- it looks rigorous and is the most likely to silently under-report.
* **Runtime coverage check** -- the driver records which columns it actually read, and the contract
  asserts the declared set covers it. Genuinely attractive, and not rejected on merit: the drivers are
  `for_each`-sharded corpus passes, so "which columns did this read" is a per-shard question with real
  plumbing behind it. Doing it now would make the cycle's central mechanism depend on new
  instrumentation rather than a proven pattern.

**Trigger for revisiting:** if PR 6 or PR 7 turns up an invalidation the mechanism missed, add the
runtime coverage check. This boundary is designed, not overlooked.

---

## 2. K9 — the output-side gate

The existing provenance gate reads driver **source**: does it import the helper, offer
`--allow-dirty`, call `require_clean_tree` from `main()`. It cannot see output. A driver can satisfy
every assertion in the registry and still emit an artifact nobody can trace.

**Measured on `main` @ 5b1a0a1, and the finding is sharper than first recorded.** An earlier note said
both bundled attempt models "carry `training_commit: null`". They do not -- the key is **absent
entirely**, and the three bundled models use **two different conventions**:

    _xshot_weights/metadata.json    (no *commit* key at all)
    _xcross_weights/metadata.json   (no *commit* key at all)
    _xshot|_xcross/metrics.json     run_commit: 6e3a132...      <- present and correct
    _ghost_gk_weights/metadata.json training_commit: 97c74d58...

An absent key is worse than a null one: a null is something a reader can notice. And the artifacts are
**not** unprovenanced -- `run_commit` in their `metrics.json` is correct -- so the defect is narrower
and more interesting than "provenance missing". The 4.74.0 retrain simply did not write the field
ghost-GK uses.

**Consequence for the gate: it cannot just assert `training_commit` is non-null.** It must first
decide WHICH field carries training provenance and apply it uniformly, because today three bundled
models answer that question two different ways. Picking one and enforcing it is part of this item, not
a precondition for it.

The new gate walks **committed artifacts**:

> every `docs/research/**/metrics.json` and every bundled `_*_weights/*/metadata.json` carries a
> non-null `run_commit`, `run_tree_dirty: false`, and -- once declared -- an `input_contract` digest.

with an explicit exemption registry, each entry carrying a reason, burned down by a sibling assertion
the way `_UNMODELLED` already is in the C4 gate.

**These ship with §1, not after it.** The source gate says what a driver must DO; the output gate says
what an artifact must CARRY. Designed apart they disagree about what counts as provenanced -- and K9
exists precisely because only one of them existed.

---

## 3. The four registry gates (items 10, 24, K2, 26)

Same shape; the care is in each derivation rule.

### 3.1 Item 10 — `ARTIFACT_DRIVERS` completeness

Today: `assert len(ARTIFACT_DRIVERS) >= 6` against **18** entries. A floor cannot detect an omission.

**The obvious derivation is circular and must be avoided.** Deriving the population from "scripts that
import `_provenance`" finds only drivers *already wired*, so `validate_xcross_causal` -- the reason
this item exists -- would never have appeared. Key on what makes something a driver, not on whether it
complies:

> a `scripts/*.py` that declares an `--out` argument AND writes a `.json` or `.md` beneath it
> (AST-detected)

**Prior art: ADR-052 already built this, and this item must reuse it rather than re-derive it.**
`tests/scripts/test_corpus_driver_resilience.py` carries `_population()` (`:153`) -- globs
`scripts/*.py`, skips underscore-prefixed files, AST-parses each, filters by predicate: structurally
the scaffolding item 10 needs. It is guarded by `test_the_pending_list_is_EXACT`, a two-directional
assertion whose docstring reads *"Fails BOTH ways -- the only thing that stops a debt list becoming a
dumping ground."*

And `_NOT_YET_MIGRATED` (`:168`) is **already the empty-bucket-as-mechanism design**, arrived at
independently below. Its comment: *"EMPTY as of ADR-052 -- every in-population driver adopts the seam.
It stays as the mechanism, not as a list: a new unmigrated driver has somewhere to be recorded WITH a
reason, and cannot arrive silently."* Cite that rationale rather than re-deriving it.

**A reconciliation assertion is required, and is a finding in its own right.** Two independent AST
walkers over `scripts/*.py` -- ADR-052's corpus-driver population and item 10's artifact-driver
population -- would otherwise drift with nothing relating them. A script may legitimately be one and
not the other, so the assertion is not equality; it is that the relationship is **stated and checked**
(e.g. every artifact driver that walks a corpus is in ADR-052's population). Without it, a change to
either derivation is silent in the other.

**Naively, assert both directions: `ARTIFACT_DRIVERS | _NOT_A_DRIVER == candidates`. Measured, that
equality CANNOT HOLD on today's tree, and no exemption registry can fix it** -- the failure is on the
left side. Three enrolled drivers are not derivable by any `--out`-keyed rule:

    calibrate_xt_bandwidth    declares --report-out   (does not start with "--out")
    train_gk_completion       declares no out-ish flag at all
    train_gk_retention        declares no out-ish flag at all

An exemption registry removes members from `candidates`; it cannot add the three that are enrolled but
underivable. So the assertion as first written fails on day one against its own registry. This is not
the "tuning against the real tree" Risk 2 anticipates -- it is the central assertion of the cycle being
unsatisfiable by construction.

**Three buckets, not two.** The gate asserts:

    candidates == (ARTIFACT_DRIVERS - _UNDERIVABLE) | _NOT_A_DRIVER

* `_NOT_A_DRIVER` -- matched the rule, correctly excluded (e.g. `render_sb360_matrix`); reason required.
* `_UNDERIVABLE` -- genuinely a driver, enrolled, but invisible to the rule; reason required, and the
  reason must say WHY it is invisible so the entry can be retired if the rule improves.

Both buckets are burned down by sibling assertions when their scripts disappear, the way `_UNMODELLED`
already is in the C4 gate.

**But three buckets alone leave a blind spot, and the measurement already names its class.** A script
that is neither derivable NOR enrolled is absent from `candidates`, from `ARTIFACT_DRIVERS`, and from
both exemption registries -- the equality still holds and the gate cannot see it. That is not
hypothetical: two of the three `_UNDERIVABLE` entries are **trainers**, invisible to any `--out`-keyed
rule because they name their destination differently. So "trainers are invisible to this rule" is
established fact, and the next trainer added to `scripts/` would be silently uncovered by the very gate
built to prevent that.

Note the asymmetry with the motivating case: `validate_xcross_causal` was unenrolled but **was**
derivable, which is why the rule reaches it. An unenrolled trainer is a different shape.

**So broaden the key until the bucket is EMPTY, and assert that it is.** Two clauses, measured:

    rule as written (--out prefix)      _UNDERIVABLE = 3   calibrate_xt_bandwidth,
                                                           train_gk_completion, train_gk_retention
    BROADENED (--*out* OR _weights)     _UNDERIVABLE = 0

* any `--*out*` flag (recovers `--report-out`, `--output-dir`)
* writes into a bundled weights path -- a `_weights` token in source (recovers both trainers, and
  `train_ghost_gk`)

**`_UNDERIVABLE` stays in the design as the right shape for a future exception, but must be EMPTY on
landing, with a sibling assertion that it is.** That is what closes the blind spot rather than
documenting it: if every enrolled driver is mechanically derivable, then a driver that is neither
derivable nor enrolled is no longer a reachable state for the classes we have -- and the day the
bucket becomes non-empty, the assertion says so.

**Verified against one known case, and CONSTRAINED by a counter-example.**

*The case it must catch:* `validate_xcross_causal` (4.74.0) had `--out`, wrote `metrics.json`, was
absent from the tuple, and its artifact carried no provenance at all. The rule flags it.

*The counter-example it must NOT flag:* `render_sb360_matrix` (4.75.0) also has `--out` and also
writes a document beneath it -- **and is correctly excluded**, with the reason recorded inline at
`tests/scripts/test_provenance_wiring.py:31`:

> "NOT enrolled: `render_sb360_matrix`, which reads a COMMITTED registry and writes a document. It
> does no corpus work and consumes no external data, so the guard would add nothing and would make the
> report unrenderable during the session that produces it."

That second reason is the sharp one, and it generalises: a guarded driver **cannot run on the dirty
tree that produces its own inputs**, so guarding a pure renderer makes it unusable at exactly the
moment it is needed. PR 5 hit that constraint four times and had to split commits around it.

**So `--out` + writes-a-file is not the discriminator on its own.** The distinguishing property is the
one their comment names: **does the script consume something outside the repository?** A corpus, the
network, a DGX path -- as opposed to reading committed inputs and rendering them.

The exclusions stay a **reasoned registry** rather than a smarter matcher. An AST rule that tried to
infer "consumes external data" would re-create the caller-sweep failure -- a syntactic test that cannot
see provenance -- which §1.4 rejects for the contract mechanism on identical grounds, and which ADR-043
acted on when it deleted the id-compat lint in favour of enumeration. Derive the population; justify
the exclusions.

### 3.2 Item 24 — the C4 aggregator count

`docs/c4/architecture.dsl:23` says **"32 action-coupled aggregators"**; zero tests reference it.

There are **two correct numbers**, and picking wrong is the likely failure:

    33  registered add_*                 the ADR-051 mirror-registry surface
    32  action-coupled aggregators       what the C4 DSL describes
         difference: add_gradientsports_player_ids, a jersey helper

The gate derives from `tracking.__all__`, carries an explicit `_NOT_ACTION_COUPLED` set with a
justification per entry, and asserts the DSL string matches. **It must name the other quantity and say
why they differ** -- otherwise the next maintainer resolves the ambiguity by making the DSL quote 33,
which turns a true sentence false in a way no test catches.

### 3.3 K2 — the ADR-020 dup-`action_id` gate

Two independent defects:

* It enumerates `dir(F)` over `tracking.features` only, so the **atomic mirrors have never been
  covered**. Extend to both surfaces. **Sized correctly: 22, not 15.** The 15 in an earlier revision was
  borrowed from CLAUDE.md's ADR-033 sentence, which counts `add_*` mirrors -- a different surface from
  the `_xfns` factories this gate governs. Measured on `silly_kicks.atomic.tracking.features`:
  `__all__` carries **22** `_xfns` names (and 18 `add_*`), while `dir()` yields 23. Building the
  replacement meta-assertion against a 15-element expectation would encode the wrong denominator.
* Its meta-assertion is
  `assert set(_XFNS_NAMES) == {n for n in dir(F) if n.endswith("_xfns")}` -- the same expression on
  both sides, always true.
* **A third defect, in the same three lines:** `assert len(_XFNS_NAMES) >= 21  # bumped for xt_gk_xfns`.
  A floor, sitting inside the very test this item repairs, with a comment recording that it has already
  been hand-bumped once. §3.1's whole argument is that a floor cannot detect an omission.

**The independent source must NOT be ADR-033's purity gate.** An earlier revision of this section said
to compare against "the `__all__` union ADR-033's purity gate already uses". That gate registers
`add_*` functions -- measured, `tests/test_add_star_purity.py` contains **zero** occurrences of
`_xfns`. Validating an `_xfns` registry against an `add_*` surface compares different populations with
different sizes; it would fail immediately and for no useful reason.

**The workable pair is `dir()` versus `__all__` on the same module** -- runtime namespace against
declared exports, two genuinely independent derivations. Measured on `silly_kicks.tracking.features`:

    dir(F)      28 names ending in _xfns
    F.__all__   24
    disagree     structural_pass_xfns, xcross_attempt_xfns, xshot_occurrence_xfns, xt_gk_xfns

That four-way disagreement is exactly the signal an independent source exists to produce, and it is
live today. Whether those four are export omissions or deliberate is a question the gate should force
someone to answer.

### 3.4 Item 26 — the NaN-safety and leakage-guard floors

Found by boundary sweep, not by reading: the genus has an instance nobody named, and it guards a
contract CLAUDE.md states as hard for the whole public enrichment family.

Measured `__all__` references per gate:

    tests/test_enrichment_nan_safety.py     3 floors,  0 __all__ refs   <- unpinned
    tests/test_add_star_purity.py           0 floors,  9 __all__ refs   <- pinned (ADR-033)
    tests/tracking/test_mirror_registry.py  0 floors,  2 __all__ refs   <- pinned (ADR-051)

ADR-003's registry is auto-discovered from the `@nan_safe_enrichment` decorator, so it is complete
over DECORATED helpers -- but **decoration is the human-maintained opt-in**, and nothing ties it to
`__all__`. A new public `add_*` shipped without the decorator is invisible: `STD_ENRICHMENTS >= 5`,
`TRACKING_ENRICHMENTS >= 10` and `ATOMIC_ENRICHMENTS >= 5` all still pass, and its NaN-safety is never
exercised. ADR-033 and ADR-051 both pin their surface to the public export in both directions; ADR-003
does not.

The floors themselves are legitimate -- they guard against discovery collapse ("Did the marker name
change or a helper lose its decoration?") -- so this **adds** a two-directional pin rather than
replacing them.

**Three more, verified during this review rather than assumed:** the leakage guards
`test_packing_xfns_leakage_guard.py`, `test_run_value_xfns_leakage_guard.py` and
`test_shot_goalmouth_no_xfns_guard.py` each carry a `>= 10` floor with **zero** `__all__` references
-- floors alone, not floors beside an exact assertion. They protect the rule CLAUDE.md calls a
HybridVAEP-class correctness break.

**Why it belongs here rather than in Cycle C:** the repair is the same one-line pattern as items 10,
24 and K2 -- add a two-directional pin beside an existing floor -- so it lands in the same phase,
against the same reviewer's attention, at roughly a quarter of that phase's cost. Splitting one
instance of a genus away from the other four would be the arbitrary choice, not keeping it.

### 3.5 All four land RED first

Observed failing against today's code before any repair. Non-negotiable here: a gate written after its
own fix has never been seen to work, and three of these four exist *because* their predecessors
were never observed failing. Item 26's own red is the two-directional pin, not the floors -- the
floors pass today and are kept.

---

## 4. Item 23 — the GS input-convention guard

The converter warns `declared=per_period_absolute but detector inferred possession_perspective` on the
pining loader path. K6 established by field measurement that the output is **correct** -- action vs
re-projected frame ball, median `|dy|` 2.75 / 2.79 m across 2,742 linked actions, with no period,
flip, or home/away split, against a calibrated scale of ~0.2 m (correct) and ~11.8 m (y-inverted).

**The defect is that the guard cannot see the case it governs.** `on_mismatch=None` resolves to
`"raise"` under `SILLY_KICKS_ASSERT_INVARIANTS=1`, which `.github/workflows/ci.yml:58` sets -- but the
committed GS fixture is `synthetic_match.json`, and real GS data is owner-tier, flowing only on the
pining path where the variable is unset. **Hard-fail where the condition cannot arise; soft-warn where
it does.**

### Order of work: measure -> shape -> diagnose

| | |
|---|---|
| **2a** (starts day one) | Measure the real per-`(match, team, period)` shot AND group distribution on owner-tier GS. Only summary counts travel, never coordinates. **Commit it as a provenanced artifact.** |
| **1** | Reshape the fixture to 2a's numbers so CI can see the case. |
| **2b** | Diagnose, and let the diagnosis choose the side. |
| **3** | Plant a genuinely mis-declared provider fixture and observe the guard fire. |
| **4** | Any GS exemption goes in a registry with a justification and a dedicated test. Never `filterwarnings`, never a per-provider `silent`. |

**2a's provenance requirement is why this item belongs in THIS cycle.** GS is owner-tier, so unlike
the `tests/datasets/sportec/idsse_slice/` precedent -- a reduced slice of real data with a
`SOURCE_SHA` -- only *statistics* can travel. A fixture shaped to an unrecorded number rebuilds the
exact failure this cycle removes.

**The binding constraint is measured, and it is not what an earlier revision assumed.** The committed
fixture has 10 shots in `(team 100, period 1)` -- AT the `high` threshold, not in the defer band --
and **only one team has shots at all**, so it defers on the *fewer-than-two-reliable-groups* clause
(`orientation.py:289-292`, `:311-321`). Give a second team or period a reliable group. Raising
per-group shot counts, which the earlier prescription called for, would not have made CI see the case.

**Do not repair the symptom by weakening the detector.** That converts a working guard into a
decorative one, which is how a real disagreement would later be lost.

---

## 5. Phasing

| phase | items | why here |
|---|---|---|
| **1** | 10, 24, K2, 26 | small, independent, all the same shape; proves the derive-the-population pattern before the mechanism depends on it |
| **2** | 9 + K9 | one build -- source-side contract and output-side check must agree |
| **3** | 23 | last in the build, but its **2a measurement runs as early as the clean-tree rule allows** -- see below |

**"Day one" is not executable, and the reason is this repo's most-hit trap.** §4 requires 2a's output to
be a provenanced artifact, so its driver calls `require_clean_tree` -- and `scripts/_provenance.py:73-75`
counts **untracked** files as dirty on purpose. On day one the tree holds this spec, the plan, and the
newly written 2a driver, all untracked, so the driver would `SystemExit` before doing any corpus work.

**Ordering, stated so it is not rediscovered as a crash:** commit the spec, the plan and the 2a driver
FIRST, then run 2a. This is the same constraint that forced PR 5 into six commits -- a driver cannot run
in the commit that introduces it.

**And 2a's driver is the first customer of the item-10 gate this cycle builds** -- a genuine self-test
worth naming as one. It also interacts with §3.1: if it names its output anything other than an
`--*out*` flag or a weights path, it lands straight in `_UNDERIVABLE` and breaks the empty-bucket
assertion. Give it `--out`.

Item 23 splits into its own PR **if it grows -- but only DOWNSTREAM of phases 1 and 2, never before or
in parallel.** Its 2a step produces a committed provenanced artifact, which makes 2a's driver a new
artifact driver: a customer of item 10's completeness gate (phase 1) and of K9's output gate (phase 2).
Splitting it earlier would have it consume gates that do not exist yet. It is small as specified and
must land before PR 6 regardless, so keeping it here is the shortest path.

---

## 6. Acceptance

* **Every gate that CAN fail today, observed failing before its repair** -- recorded rather than
  asserted afterwards. **Scoped deliberately:** K9's gate has two halves, and its
  `docs/research/**/metrics.json` half lands **GREEN by construction** -- measured, 7 of 7 artifacts
  carry a `run_commit` and none is dirty, because K3's offender was repaired in 4.74.0 and nothing has
  replaced it. Only the `_*_weights/*/metadata.json` half lands red, on the two bundled models §2
  identifies. A green half is a real signal (the research-artifact corpus is currently clean) but it is
  not RED-first evidence, and an unqualified "every gate observed failing" would invite manufacturing a
  failure to tick the box.
* **Item 10's rule verified against the known case** -- it must flag `validate_xcross_causal` when
  un-enrolled.
* **K9's output gate run against today's artifacts, and the criterion stated in §2's terms.** An
  earlier revision required "`training_commit: null` surfacing as a real finding" -- but §2 established
  by measurement that **no artifact carries a null**; the key is absent. That criterion is satisfiable
  by vacuity: nothing surfaces, the box is ticked, and the actual defect goes unexamined. Inside the
  acceptance section of the cycle built to remove exactly that failure. The criterion is: **the gate
  names which field carries training provenance, applies it uniformly, and surfaces the two bundled
  models that carry no such key** while ghost-GK carries `training_commit`.
* **The contract mechanism exercised against a real change, not a hypothetical.** PR 6 is the first
  consumer: its invalidation list should be produced by the mechanism, not by hand. If the mechanism
  misses something PR 6 turns up, that is the trigger for §1.4's runtime coverage check.
* Full suite 0 failed; ruff + `ruff format --check` + pyright clean at CI scope
  (`silly_kicks/ tests/ scripts/`, never `.`); C4 gates pass.
* `--merge`, never squash, if any artifact in the PR **stamps OR cites** a commit SHA. The narrower
  "stamps a `run_commit`" wording would have missed 4.75.0, whose binding citation was a SHA in a
  markdown provenance table (`docs/research/sb360_coverage/coverage.md`), not a JSON field.

## 7. Risks

1. **Under-declaration** (§1.4) -- the designed limit. Mitigated by symbol-referencing declarations and
   by a stated trigger for escalating to runtime coverage.
2. **`_UNDERIVABLE` does not stay empty as `scripts/` grows.** The rule was measured against the real
   tree and the buckets balance today (candidates 19; `(18 - 0) | _NOT_A_DRIVER 4` -- see §3.1), so the
   open question is no longer "will it need tuning". It is whether a future driver arrives in a shape
   neither clause reaches. The sibling assertion makes that loud on the day it happens rather than
   silent, which is the property that matters; the cost is a real one-line decision each time, not an
   invisible gap.
3. **Item 23's 2a measurement may be inconclusive** -- if the real distribution sits near a band
   boundary, "shape the fixture to it" underdetermines the fixture. Then state the choice and its
   reason in the fixture generator rather than picking silently.
4. **This spec can be incomplete in the way its predecessor was.** PR 5's spec was, twice, within a
   day. Anything discovered outside §Scope is recorded as a failure of this document rather than
   absorbed.
