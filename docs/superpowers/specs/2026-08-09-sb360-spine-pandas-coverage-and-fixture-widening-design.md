# SB360 spine: pandas coverage, the snapshot dtype question, and reclaiming audit coverage — Design

**Status:** IMPLEMENTED as 4.79.0 — and the cycle grew well beyond this document. See the CHANGELOG.
**Branch:** `sb360-pandas-coverage-and-fixture-widening` (drafted off `main` @ `a29ae0f` / 4.77.1;
rebased onto `e0d69cd` / 4.78.0 when the parallel Cycle B landed).
**Predecessors:** ADR-053 (SB360 audit), ADR-054 (degradation + parse port), ADR-055 (goal-map seam +
visible-area). This cycle carries no new ADR unless §7 says otherwise.

> **ANNOTATED, NOT REWRITTEN.** What follows is the design as APPROVED, kept as evidence of what was
> decided up front. Retconning it to match the outcome would destroy the only record of the
> difference — and the difference is the interesting part. Three claims below are now false, and
> each is flagged where it appears rather than silently corrected:
>
> 1. **§7 "No new ADR is expected"** — the cycle produced TWO: **ADR-057** (the pandas span, which
>    §7's own criterion selected) and **ADR-058** (tracking-frame id dtypes are nullable), which
>    §7b explicitly deferred as "not this cycle".
> 2. **§7b "Not this cycle — it moves goldens and touches every adapter"** — done here. The owner
>    declined the deferral. It moved NO goldens; the "touches every adapter" fear was wrong because
>    every adapter already overrode the base.
> 3. **§3 Item 4's fix-list framing** — the discriminator convicted **0**, not the 2 the interim
>    handoff recorded. Its first two revisions measured the wrong quantity; see the CHANGELOG.
>
> Three items the cycle delivered that this document never scoped: the `add_xcross_attempt`
> velocity contract (it had NO audit coverage at all), the `add_elastic_sync` empty-distance-lookup
> defect (found only because §7b's deferral was overturned), and two standing CI gates replacing
> what would otherwise have been `TODO.md` rows.

## 1. Why this cycle exists

The 4.77.0/4.77.1 cycle deferred work and — when the owner asked *"what else has been deferred?"* —
three items turned out to be tracked in a test pin, an ADR and code comments but **not** in
`TODO.md`, the only place an owner looks. This cycle folds the eligible ones back in, grouped with
the remaining SB360 work they naturally sit beside.

**Ordering constraint.** A second session is mid-flight on `cycleb-artifact-contracts`, whose pushed
branch touches `silly_kicks/tracking/_kernels.py`, `utils.py`, `atomic/tracking/features.py` and
`tests/scripts/`. **Every item below is disjoint from that surface** (measured, §6). The D3 re-key,
which is *not* disjoint, is deliberately held for a checkpoint decision on this same branch after
these land.

## 2. The finding that reshaped the cycle

`TODO.md` recorded, of the `snapshot_to_tracking_frames` dtype question:

> The concern is only checkable on a pandas-3 environment, which CI does not have
> (`ci.yml` is OS x Python only). Next step is that environment, not another blind fix.

**That is false.** Measured on CI run `31316804815` (`main` @ `a29ae0f`), by extracting the resolved
version from each leg's install step:

| CI leg | pandas |
|---|---|
| `ubuntu-latest, 3.10` | **2.3.3** |
| `ubuntu-latest, 3.11` | **3.0.5** |
| `ubuntu-latest, 3.12` (primary) | **3.0.5** |
| `windows-latest, 3.12` | **3.0.5** |
| `lint` | 3.0.5 |

**Three of four test legs already run pandas 3.** The row confused *"no pandas AXIS"* with *"no
pandas-3 environment"*: `pyproject.toml` pins `pandas>=2.1.1,!=3.0.4` — deliberately permitting
pandas 3 — and with no upper bound pip resolves the newest compatible version per interpreter, while
pandas 3 requires Python ≥3.11. The differential coverage the row asked for is already present and
already free.

**But it is ACCIDENTAL, and that is the real defect.** Nothing declares or asserts it. When pandas
3.1 ships, or when 3.10 support is dropped, CI silently changes what it tests with no diff and no
signal. This repo already has one measured instance of that class — DAS going silently all-NaN on
pandas 3 — so the hazard is demonstrated, not hypothetical.

## 3. Scope — four items, in dependency order

### Item 1 — Make CI's pandas span explicit and asserted

**Problem.** The pandas major each leg resolves is invisible and unpinned. Coverage across both
majors is real today and could vanish without a diff.

**Design.** A guard test asserting the matrix **spans both pandas majors**, deriving the resolved
version rather than hard-coding it, in the mould of `tests/test_ci_slow_gating_wired.py` and
`tests/test_ci_publish_guard_wired.py` (structural assertions over `ci.yml`, YAML-parsed, not
substring-matched).

Two sub-decisions, both deliberate:

- **Assert the SPAN, not specific versions.** Pinning "3.10 → pandas 2.3.3" makes the guard fail on
  every routine dependency bump — noise that trains a reader to edit the expectation without
  thinking. Asserting *at least one leg on pandas 2 and at least one on pandas 3* fails only when
  the property that matters is actually lost.
- **The resolved version must be OBSERVED where it can be, not inferred from `pyproject.toml`.**
  Reading the constraint re-asserts the input; what matters is what pip actually installed.

**Feasibility note — a test cannot see other legs.** A test running in one leg observes only its own
interpreter, so no single test can assert "the matrix spans both majors" by observation alone. The
guard is therefore a PAIR, and neither half is sufficient:

1. **Observed, across legs:** each test leg writes its resolved pandas major to an artifact, and a
   `needs: test` job asserts the UNION spans both majors. This is the only place in the system where
   the span is actually observable, and it is ~15 lines of workflow.
2. **Structural, over `ci.yml`:** assert the **resolved LEG SET** still straddles the Python 3.11
   boundary that makes two majors reachable at all, given `pandas>=2.1.1,!=3.0.4` with no upper
   bound.

**Half (2) must parse the leg set, NOT the `python-version` axis.** `ci.yml` computes legs as
os × python-version **minus `exclude` plus `include`**, and `exclude` is already the pruning
mechanism in use (two windows legs). Adding `{os: ubuntu-latest, python-version: "3.10"}` to
`exclude` collapses the pandas-2 span while leaving `"3.10"` in the axis — an axis-based assertion
**passes**, and that is the likelier edit than deleting an axis entry. `tests/test_ci_slow_gating_wired.py`
already reads `matrix.get("include", [])` rather than trusting the axes; follow it.

An earlier draft specified half (1) as *"assert the pandas major actually imported, and record it"*.
That is **not a gate**: `pandas.__version__.major in (2, 3)` is tautological, asserting a specific
major per interpreter reintroduces the hardcoding this item rejects, and "record it" had no sink —
so nothing anywhere observed the span, which is the entire property. Hence the artifact + aggregation
job above: it has a real failure condition (the union covers one major) and a real reader.

**Write (2) so its failure message states the boundary assumption**, or a future reader will "fix"
it by moving the boundary rather than restoring the leg.

**Non-vacuity requirements.**
- Half (2) red against a `ci.yml` whose pandas-2 leg is removed **via `exclude`**, not by deleting
  the axis entry — the mutation must match the hazard, not the implementation.
- Half (1) red against a recorded union containing a single major.

**Also in this item:** correct the false `TODO.md` row, quoting the measured table in §2.

### Item 2 — The `snapshot_to_tracking_frames` dtype differential test

**Problem.** 4.77.0 planned a dtype pin, then dropped it (ADR-055): casting to
`TRACKING_FRAMES_COLUMNS` is unimplementable because it declares `int64` for `player_id`/`team_id`,
the ball row is NA in both, and `int64` cannot hold NA (`IntCastingNaNError` on every snapshot); a
`restore_id_dtype`-based pin was measured to change nothing (0 of 2 tests written for it went red).
The row's remaining claim — that the behaviour may differ across pandas majors — was never checked.

**Design.** A test that pins what `snapshot_to_tracking_frames` produces for `player_id`/`team_id`
across the input dtypes that matter (numpy int, nullable `Int64`, object), asserting the property
that consumers actually depend on rather than a literal dtype name. Per ADR-019 the contract that
matters downstream is that **`id_compat` comparisons keep working**, not that a specific dtype
appears — so the assertion is behavioural: ids built from the snapshot must compare equal to the
same ids in their source form, on whichever pandas the leg resolves.

**This is why Item 1 comes first:** the test's value is that it runs on both majors, and Item 1 is
what stops that silently ceasing to be true.

**Expected outcome is a measurement, not a fix.** Measured on 2.3.3 during 4.77.0, the concat yields
`float64` for the as-built numeric-int fixture and an `Int64` source stays `Int64`. If pandas 3
agrees, the deliverable is the test plus a corrected TODO row. **Do not pre-commit to a repair for a
divergence not yet observed.**

**What LANDS on divergence, decided now rather than improvised under CI pressure.** Three of four
legs run pandas 3, so a divergence means a red test on a branch that cannot merge. In that case
land the test as `xfail(strict=True)` whose message carries the measured behaviour of BOTH majors,
plus a `TODO.md` row scoping the repair. Strict, so the marker must be deleted when the behaviour is
fixed rather than rotting into an exemption. That closes the cycle with the measurement preserved,
which is the deliverable this item is for.

### Item 3 — `gk_one_end`: reclaim audit coverage by ADDING a roster, not widening one

**Problem.** `NOT_EXERCISED_BUDGET` rose 26 → 31 in 4.77.1. `gk_absent` removes BOTH keepers, so
`resolve_defended_goals` falls to its outfield rung and guesses both teams at x=105 (measured
outfield mean x 56.9 and 76.5, both past the 52.5 midline). A both-teams-same-end map is DEGENERATE,
`attacked_goal` refuses it by its documented same-end guard, and `add_cover_shadows` emits NaN on
both legs for the same roster-driven reason — so no informative row survives.

**Design.** Add a THIRD visibility roster `gk_one_end`: one keeper visible, the other off-frame.

`gk_absent` is **not** an artificial degenerate case — it is a visibility axis testing a real SB360
condition (keeper outside the camera's observed region), and it is the only case exercising the
both-absent refusal path. Widening it would trade one coverage loss for another. `gk_one_end` adds
coverage instead: the visible keeper's team RESOLVES, the other falls to the outfield rung, the two
ends differ, the map is non-degenerate, and the five cover-shadow columns become exercisable.

It is also the better-supported case. Per the committed coverage report the DEFENDING keeper is
in-frame on **92.2%** of shots, so a freeze-frame with a keeper present is the common one and
`gk_absent` alone leaves that majority shape unexercised.

**Correction, made during implementation.** This paragraph originally read *"the defending keeper is
in-frame 92.2% of the time on shots while the acting-side keeper usually is not"*. The second clause
is not measured: `coverage.md`'s `acting GK` cell for `shot` is `—`, which the table's own legend
defines as **definitionally zero / not applicable** (the keeper is not the actor on a shot), not a
low observed rate. The report says nothing about the far keeper's presence. The roster's
justification stands on the 92.2% alone, and the comparative was removed rather than re-sourced.

**Cost, stated plainly:** a new roster axis multiplies the verdict registry — every registered
`add_*` gains a verdict under `gk_one_end` requiring machine observation (CI-re-derived) and human
adjudication (with rationale, per ADR-053). That is the price of the coverage and should be paid
knowingly.

**The budget CANNOT drop, and an earlier draft of this spec asked for exactly that.** Verified:
`NOT_EXERCISED_BUDGET` counts `(entry, axis, roster, column)` tuples — `test_registry_surface.py:158`
asserts **equality** over `iter_verdicts`, which yields `(_axis, _roster, _col, v)` — and the
registry holds **35 entries, each declaring one verdict block per roster** (35 `gk_absent`, 35
`defender_absent`, measured). A third roster therefore ADDS a column to that matrix; it cannot
remove tuples from the `gk_absent` column, and the requirement below forbids touching them. So the
count can only **rise or stay flat**. Requiring a drop while forbidding a `gk_absent` rebaseline is
self-contradictory, and the implementer would have hit it on the first CI re-derivation.

**The metric never matched the claim.** What this item delivers is *"these five columns are now
exercised SOMEWHERE"* — a per-column property across the roster sweep — not a reduction in
per-roster tuples. Both are recorded, and they are different numbers.

**Requirements.**
- The new roster must be built by the SAME `_fixture.py` path as the existing two — a bespoke
  fixture would test the fixture, not the library.
- **The five `add_cover_shadows` columns must carry a non-`not_exercised` verdict under
  `gk_one_end`**, named individually in the cycle's record. This is the coverage claim, and it is
  the assertion that makes the item worth doing.
- **`NOT_EXERCISED_BUDGET`'s rise must be bounded, enumerated per-tuple, and justified** — which is
  what its own docstring already demands (*"Raised only with a recorded reason; it is a budget, not
  a tally"*). A rise with a per-tuple reason is acceptable; an unexplained rise is not.
- **The success criterion is the named-five assertion above, promoted into
  `test_registry_surface.py` as a real test** — not a task note, and not a new aggregate metric.
  **A second draft of this spec proposed `columns_exercised_on_no_roster` (columns
  `not_exercised` under EVERY roster) and claimed it "drops by five". Measured: it does not move at
  all.** The five columns are `honest_nan` under `defender_absent` (`_entries/_space.py:96-101`), so
  they were never in that set; the set today is exactly four members
  (`add_cover_shadows.max_single_defender_player_id`, both `add_press_commitment` columns, and
  `add_xshot_occurrence.xshot_occurrence`). *"Unexercised everywhere"* is a strictly STRONGER
  predicate than *"unexercised on the roster in question"*, and the deliverable is the weaker,
  roster-specific one. **That is the same error twice** — the first metric was impossible because of
  its counting unit, the second because of its predicate strength — and both times the metric was
  invented to match the claim instead of the claim being read off what changes.
- **`columns_exercised_on_no_roster` is still worth building, as a standing regression pin** — it
  catches a column going dark on every roster, which nothing else does. It will register **zero
  change** from this cycle, and that is the correct expectation, not a disappointment.
- **`gk_absent`'s existing verdicts must be PINNED, not merely expected to hold.** Capture the
  `gk_absent` slice before the change and assert it byte-identical after. Verdicts are re-derived by
  CI, so a leak into the wrong axis would otherwise be absorbed into the new baseline silently —
  the same silent-rebaseline shape this repo has already shipped once.

### Item 4 — The velocity-fixture sweep, behaviourally discriminated

**Problem.** ADR-053/4.76.0 found two fixtures declaring `speed_source="native"` with no `vx`/`vy`,
so a fitted model scored on 5-of-26 imputed features. The ghost path now REFUSES that input, but
every other velocity consumer stays silent.

**Measured candidate set (crude grep, an UPPER BOUND).** 65 test files reference `speed_source`; of
those, **46 declare `native`/`derived`**, and **24** of those supply no `vx`/`vy`. The three numbers
have different denominators and a reviewer independently got 65/29 against 46/24 by counting the
first and third differently — so the **command is the spec, not the number**:

```bash
# denominator A: files referencing speed_source at all
grep -rlE 'speed_source' tests --include="*.py" | wc -l
# denominators B (claims) and C (claims with no vx/vy): re-derive with
python scripts/… # the discriminator itself reports A, B, C in its output (step 4)
```

Since the discriminator reports these counts as part of its own output, the spec deliberately does
not pin them: a number in prose with a half-life of days is exactly what this cycle is correcting
elsewhere.

**24 is not a defect count and must not be treated as one.** A fixture that never reaches a velocity
consumer is correct as written; "fixing" 24 files would churn the suite and bury the real cases.
This repo has already recorded that keyword/substring tests over source are not evidence of
behaviour.

**Design.** Build the discriminator first, then let it produce the fix-list:

1. For each candidate, determine whether its frames actually reach a velocity consumer.
2. For those that do, determine whether the consumer's output CHANGES when velocity is supplied —
   the 4.76.0 defect signature (imputation silently altering a scored value).
3. Fix only the fixtures that fail (2), each with a test that would have caught it.
4. Report the discriminator's own numbers: candidates, reached-a-consumer, value-changed, fixed.

**Step 2 must supply a PERTURBING velocity, and record the delta.** `vx = vy = 0` produces no change
in many consumers even where imputation genuinely matters, so "value didn't change" becomes
indistinguishable from "the probe didn't perturb anything" — and this repo has already named
`vx=vy=0` as a fixture defect, not a convenience. The applicability axis already solved this shape
(`tests/sb360/_registry.py`: *"col -> {extreme: delta, near: delta}. Recorded so a zero-movement
classification is VISIBLE: a `no_support` derived from two zero deltas is indistinguishable from a
probe that silently failed to perturb anything"*). **Reuse it: record the probe deltas beside each
verdict**, in the same report §3 already requires, so a "no change" outcome is auditable rather than
asserted.

**A fixture that claims velocity, reaches a consumer, and whose value does NOT change is left
alone** — and that outcome is recorded rather than silently dropped, because "we checked and it
didn't matter" is a finding.

## 4. What is deliberately NOT in this cycle

| Item | Why |
|---|---|
| **D3 re-key** (`compute_defensive_line`, `_packing`) | Not disjoint from the other session's surface. Held for a checkpoint on this branch. **Whoever does it must add a Gate C entry in the same change** — Gate B is their only detector today and goes vacuous the moment either is re-keyed (ADR-055). |
| **`sportec_slim.parquet` mirror repair** | Repair moves `sportec_expected.parquet` and the lakehouse-parity goldens. Its own change; the strict xfail already forces the marker's deletion on repair. |
| **Wiring `visible_area` into count features** | ADR-009: changes existing values AND decides for the consumer what a partial observation means. Needs a consumer asking, not a library decision. |
| **Lakehouse StatsBomb adoption** | Not answerable from this repo. |
| **`render_sb360_matrix.py` exemption** | Depends on an `ARTIFACT_DRIVERS` completeness gate that has not landed. |

## 5. Testing and gates

Every new gate in this cycle must be **observed RED** against a reintroduced defect before it is
accepted — landed-red where practical, mutation-verified otherwise. This is the ADR-051 rule and it
caught real vacuity twice in the previous cycle.

Specifically:
- Item 1 half (2): red against a `ci.yml` whose pandas-2 leg is removed **via `exclude`**.
- Item 1 half (1): red against a recorded union containing a single major.
- Item 2's test: red against a **mutated `snapshot_to_tracking_frames`** that breaks the id-identity
  property — e.g. **dropping the ball row's NA**, so that row's id stops comparing equal to its
  source. An earlier draft also offered "stringify the ids so `id_compat` still matches": that is a
  mutation the test **survives by design**, since the assertion is that `id_compat` comparisons keep
  working. Offering it would have cost an hour and then tempted the implementer to weaken the
  assertion back toward a dtype literal — the exact trap that made the original row unverifiable for
  two cycles. Deleted.
- Item 4's DISCRIMINATOR (not just its fixes): the two fixtures ADR-053/4.76.0 already identified —
  `test_ghost_gk_orientation.py` and `test_action_ltr_mirror_invariance.py`, which declared
  `speed_source="native"` with no `vx`/`vy` and reached a scored model on 5-of-26 imputed features —
  are a free **positive control**. Run the discriminator against their pre-4.76.0 shape; **if it
  does not surface both, the instrument is broken.** Without this, a discriminator that reports zero
  reached-a-consumer produces a false all-clear indistinguishable from a real one, and that gets
  written into `TODO.md` as a finding.
- Item 3: **the success criterion** is that the five `add_cover_shadows` columns carry a
  non-`not_exercised` verdict under `gk_one_end`, named individually, asserted in
  `test_registry_surface.py`. `NOT_EXERCISED_BUDGET`'s rise is enumerated per-tuple with a reason.
  `columns_exercised_on_no_roster` is a standing pin expected to register **zero change**.
  **Neither aggregate can express the deliverable** — the budget counts per-roster tuples so a third
  roster can only add to it, and the no-roster set already excludes the five because they are
  `honest_nan` under `defender_absent` (§3 Item 3).
- Item 4: each fixture fix ships with a test that fails on the pre-fix fixture.

Standard gates unchanged: full suite `-m "not e2e"`, pyright, ruff at CI scope
(`silly_kicks/ tests/ scripts/`).

## 6. Conflict surface (measured)

**Run this rather than trusting the number below** — it moved from 25 to 35 files within hours of
this spec being written, and a "measured" claim with a half-life of hours belongs in a command:

```bash
git fetch origin --quiet
git diff --name-only origin/main...origin/cycleb-artifact-contracts \
  | grep -E '^\.github/workflows/ci\.yml$|^tests/sb360/|test_snapshot'   # must print NOTHING
```

At time of writing that diff is **35 files** and the intersection is **empty**. The one adjacency is
`tests/tracking/test_packing_xfns_leakage_guard.py`, which the other session touches and this cycle
does not — it belongs to the D3 work held back in §4.

This sees only their PUSHED branch; they may hold uncommitted work. Re-run before merging.

## 7. ADR

> **FALSE IN HINDSIGHT (see the header annotation).** Two ADRs shipped: **ADR-057** (pandas
> span) and **ADR-058** (nullable id dtypes, which §7b below deferred). The criterion in this
> section worked — it is the PREDICTION that was wrong.

No new ADR is expected. Items 2–4 execute or correct decisions already recorded in ADR-053/054/055.
**Item 1 may warrant one** if the pandas-span guard is judged a cross-cutting CI contract rather
than a local guard.

**That decision gets its own plan step, immediately after Item 1's guard is working.** A deferred
decision with no owner does not get made — it gets discovered at commit-prep, when the cheap moment
to write an ADR has passed. The criterion: if the guard ends up asserting a property other code must
respect (a declared pandas span), it is a contract and earns an ADR; if it only pins CI's own
configuration, it is a wiring guard like `test_ci_slow_gating_wired.py` and does not.

## 7b. The boundary defect Item 2 works around (recorded, NOT fixed here)

Item 2 pins the *behaviour* consumers depend on rather than a dtype name. That is right for this
cycle, but it is a workaround for a schema defect worth naming:

`silly_kicks/tracking/schema.py` declares `player_id: int64` and `team_id: int64`. The ball row is NA
in both, so — as ADR-055 measured — the cast is **unimplementable**, raising `IntCastingNaNError` on
every snapshot. The same file already carries `KLOPPY_TRACKING_FRAMES_COLUMNS` and
`SPORTEC_TRACKING_FRAMES_COLUMNS`, which override those columns to `object`.

**A schema constant that two of its own adapters must override, and that one producer cannot satisfy
at all, is not describing a contract — it is describing one producer's happy path.** The durable fix
is `Int64` (nullable), which is true for every producer and collapses the three variant dicts.

**Not this cycle** — it moves goldens and touches every adapter. **[DONE IN THIS CYCLE — ADR-058. Both halves of this sentence were wrong: it moved NO goldens, and 'touches every adapter' was backwards — every adapter already OVERRODE the base, which is exactly why the base could change safely. Doing it then exposed the `add_elastic_sync` defect.]** Add it to `TODO.md` as its own row,
noting that it is the reason Item 2 exists.

## 8. Version

Not claimed. `main` is 4.77.1; the next number is taken at commit-prep per the standing rule, since
a parallel session may claim one first.
