# StatsBomb 360 Coverage Audit — Design Spec

**Date:** 2026-08-04
**Status:** Draft — rev 5 (post-review ×4)
**PR:** TBD (next version from `main`; tree reads `4.73.0`, so `4.74.0` — confirm at PR time)
**Scope:** Audit + registry + CI gate. No new aggregator, no library behaviour change.

## Motivation

A commercial StatsBomb 360 collaboration with an NWSL team is a realistic near-term
possibility, scoped to goalkeeper metrics. Before that data lands we need a defensible
answer to: **have we utilised SB360 everywhere it can be utilised, or are there metrics
implemented for full tracking that could be extended to freeze-frames?**

## Problem

SB360 is not unsupported. `tracking.snapshot_to_tracking_frames` (`_snapshot.py`, spec
`2026-05-27-snapshot-to-tracking-frames-design.md`) already converts per-event
freeze-frames into the 20-column tracking schema plus pre-built links, and names SB360
"Tier 2" explicitly. It is *partially* wired, and nobody knows how partially.

Three gaps are established by inspection:

1. **No producer.** `silly_kicks/providers/` contains only `sportec`. `spadl/statsbomb.py`
   has zero `freeze_frame` handling. Building the `snapshots` contract — 360 JSON →
   `action_id`/`team_id`/`is_goalkeeper`/`x`/`y`, including the 120×80 → SPADL transform —
   is entirely caller-side, per that spec's "Not in scope".

2. **The compatibility table has rotted.** The 2026-05-27 spec documents ~10 downstream
   functions under a two-state model (works / gracefully degrades). `tracking.__all__` now
   exports **33** `add_*`, and `gkdv/` and `xtgk/` (v2) landed since. Nothing failed when
   the surface grew — there was no gate to fail.

3. **Degradation is not uniform, and some of it is silent.** Exactly two consumers honour
   the velocity-availability contract: `_das.py:259` and `_press_commitment.py:100`, both
   via `velocity_unavailable_by_design`. Every other velocity-touching module handles
   absent kinematics ad hoc or not at all — several zero-fill `vx`/`vy` and carry on.

Gap 3 is the defect shape CLAUDE.md already names four times: *a plausible number from a
computation that had not happened.*

> **⚠ Corrected 2026-08-04 by execution.** Every draft of this spec through rev 5 cited
> `add_gk_influence` as the motivating example: it never consults the marker, `:205-206`
> default absent `vx`/`vy` to `0.0`, `:406-408` `np.nan_to_num` the back line — therefore it
> "likely returns numbers computed as if every player were stationary". Always labelled a
> hypothesis, and **the hypothesis is false.** Measured: `add_gk_influence` observes `all_nan`
> on all four columns, cause isolated as *velocity*. It **declines** on freeze-frames rather
> than fabricating — adjudication `honest_nan`.
>
> The zero-fill is real and *is* reached (`features.py:5219-5223` runs `frames["vx"] = 0.0`);
> something downstream still yields NaN. **A code path being reachable is not evidence about
> the value it produces**, and five review rounds accepted the inference because reading
> cannot settle it.
>
> The genuine fabrication candidates are the **pitch-control family** — `add_obso`,
> `add_pitch_control`, `add_space_creation`, `add_ghost_gk` and part of `add_pausa` — which do
> return finite numbers on zero velocity. The audit's premise survives; its example did not.

## Non-problem (checked, so a future reader does not re-open it)

`_snapshot.py:131` hardcodes `team_attacking_direction="ltr"` for every row. This is
**correct and deliberate**, documented at `_action_orientation.py:56-58`: snapshot frames
are already in SPADL action-LTR, so "never flip" is right post-ADR-028, and
`validate_period_directions` explicitly accepts a uniformly-labelled column as a
legitimate convention rather than a contradiction.

## What the audit answers

| # | Question | Evidence layer |
|---|---|---|
| Q1 | Does each aggregator *run* on freeze-frame input? | Layer A, executed |
| Q2 | When it runs, are the numbers *meaningful* or fabricated? | Layer A, paired-fixture |
| Q3 | Do real freeze-frames *contain* enough to feed it? | Layer B, real open 360 |
| Q4 | What does SB360 carry that we have no metric for? | Layer B′, `visible_area` |

Static reading answers none of them, and is the method that produced the rotted table.

## Observation vs adjudication

The core design decision. An earlier draft used a single five-valued verdict, conflating a
machine observation with a human judgement, and broke three ways: the CI gate could only
lock the key set (a repaired function kept a stale verdict while CI stayed green); one
verdict (`infeasible`) was unproducible by the decision rule; and the mechanical rule could
not distinguish *fabricated* from *legitimately different*.

### Two levels, because a raise is not a row property

An earlier draft put `row_raises` in the row table and claimed exhaustiveness. It was not
exhaustive, and the reason is a category error: **if a call raises there is no output
frame, so there are no rows to classify.** A raise is a call-level outcome that
short-circuits row classification. Mixing the levels left three of the nine
(Leg A, Leg B) ∈ {finite, NaN, raise}² cells unadmitted — and in implementation an
unadmitted pair either throws on lookup or falls through to a default, which is a silent
misclassification in the machine half of the design.

**Level 1 — call outcome.** Evaluated first; row classification runs only in the
`both_succeeded` case.

| Call outcome | Condition |
|---|---|
| `raises_a` | Leg A raised (Leg B irrelevant) |
| `raises_b` | Leg A succeeded, Leg B raised |
| `both_succeeded` | Neither raised |

`raises_b` is a real state, not a defensive stub: Leg B is a synthetic full-tracking
construction, so a fixture defect raises there specifically. It fails the audit as a
fixture bug rather than being recorded as a library property.

**Level 2 — row class.** Both calls succeeded, so each cell is
(Leg A ∈ {finite, NaN}) × (Leg B ∈ {finite, NaN}), with both-finite split by tolerance:

| Row class | Leg A | Leg B |
|---|---|---|
| `row_identical` | finite | finite, equal within tolerance |
| `row_differs` | finite | finite, not equal within tolerance |
| `row_nan_a` | NaN | finite |
| `row_nan_b` | finite | NaN |
| `row_nan_both` | NaN | NaN |

`row_nan_b` was missing. It is reachable: a feature guarding on link ambiguity sees one
linked frame in Leg A and many in Leg B, and may decline on Leg B alone.

**Exhaustiveness is a test, not a sentence.** The harness asserts every (row, column) pair
matched exactly one class, with the unmatched count and a sample of unmatched pairs in the
failure message. A claim of exhaustiveness that is not executed is the same thing as the
prose compatibility table this audit exists to replace.

### Column-level aggregation — explicit, not implied

`row_nan_both` rows are **uninformative** — neither leg said anything, so they carry no
evidence either way. They are excluded from the denominator rather than counted as
agreement. Define **informative rows** = every row that is not `row_nan_both`.

Rules are applied in this **precedence order**; the first match wins. Every state named
anywhere in this design appears here — including both Level 1 call outcomes, which an earlier
draft defined a level up and then omitted from the table that claims to be the complete
decision procedure.

The **Kind** column is three-valued, not a pass/fail binary. An earlier draft wrote "fails the
audit" against both `leg_b_declined` and `no_signal`, which are different things: the first
never reaches the registry at all, while the second reaches it, carries a mandatory rationale
and is counted against a locked budget. Collapsing them would make the entire
liveness → `not_exercised` → locked-count mechanism unreachable for anyone implementing from
the table.

| # | Observation | Rule | Kind |
|---|---|---|---|
| 1 | `raises_a` | Call outcome `raises_a` | adjudicated |
| 2 | `raises_b` | Call outcome `raises_b` | terminal fixture failure |
| 3 | `leg_b_declined` | Any `row_nan_b` | terminal fixture failure |
| 4 | `no_signal` | No informative rows | budgeted |
| 5 | `all_nan` | Every informative row `row_nan_a` | adjudicated |
| 6 | `partial_nan` | ≥1 `row_nan_a` and ≥1 other informative row | adjudicated |
| 7 | `differs` | No `row_nan_a`; ≥1 `row_differs` | adjudicated |
| 8 | `identical` | Every informative row `row_identical` | adjudicated |

Rules 1 and 2 are mutually exclusive by their Level 1 definitions (`raises_a` is "Leg A
raised, Leg B irrelevant"), so their relative order is immaterial; 2 follows 1 so the table
cannot be read as disagreeing with the level above it.

| Kind | Reaches registry? | Requirements |
|---|---|---|
| `terminal fixture failure` | no | Fixture must be repaired; never recorded as a library property |
| `budgeted` | yes | Mandatory rationale; count pre-registered and locked |
| `adjudicated` | yes | Adjudicated per the admissibility table |

Rules 5–8 are exhaustive over a non-empty informative set containing no `row_nan_b` (rule 3
having removed that case): either it contains a `row_nan_a` — all of them (5) or not (6) — or
it does not, and then it either contains a `row_differs` (7) or is entirely `row_identical`
(8). The harness asserts this rather than trusting the paragraph.

Rule 3 fires on **any** single `row_nan_b`, which is right for a deterministic synthetic
fixture — but the failure message carries the count and a sample, so one row in ten thousand
is distinguishable from systematic decline.

**Why `no_signal` exists.** An earlier rule read `identical` = "every row `row_identical`
**or** `row_nan_both`", so a column NaN in *both* legs on *every* row satisfied it and
adjudicated to `works` with no rationale. That column produced nothing anywhere; it was not
working, it was **unexercised**. Not hypothetical: on the visibility axis with the keeper
removed, a goalkeeper-specific feature is NaN in both legs by construction and would have
been recorded as `works`.

**Why the exclusion is a denominator rule, not a tightening.** The first attempt at this fix
narrowed `identical` to "every row `row_identical`" and immediately orphaned a legitimate
case: a shot-only feature is NaN in both legs on non-shot actions and identical on shots,
matching no rule at all. Excluding uninformative rows from the denominator handles the
unexercised column and the sparse-domain column with one mechanism instead of trading one
hole for another.

**Why `leg_b_declined` fails rather than adjudicates.** Leg B declining where Leg A produced
a value means the richer data yielded less output — a broken comparison for those rows, not
a library property. It is the same category as `raises_b`: a fixture-integrity failure,
surfaced as such.

The row-class tally is recorded in the report for interpretation; the registry locks the
column-level **label**, so an incidental shift in how many rows fall in each class does not
fail CI while a change in kind does.

### Per-column liveness

`no_signal` closes the vacuity hole at column level, which is where it belongs. After the
canary fix, non-vacuity otherwise rests on one live assertion plus observation-map set
equality — and **set equality guarantees stability, not meaningfulness.** Thirty dead
columns would lock as `identical` and match forever.

So: **every column must show at least one finite value in at least one leg**, per axis. A
column that cannot meet this on some axis is a fixture inadequacy, not a library finding,
and must be adjudicated `not_exercised` with a written rationale. The **count** of
`not_exercised` entries is itself pre-registered and locked, so it cannot quietly grow —
the same reasoning as ADR-052's rule that a bounded pass logs what it dropped.

### Comparison and tolerance are part of the lock

`identical` is a comparison, and the tolerance chooses the answer. Leg A and Leg B run
different code paths through pitch control, TTI and convex-hull geometry, so exact float
equality would produce spurious `differs` on semantically identical columns — each of
which then demands a human adjudication and rationale, filling the audit with noise.

More seriously: **loosening a tolerance converts `differs` into `identical`, which
manufactures a `works` verdict.** The tolerance is therefore registry state, not a harness
detail.

- Default: `rtol=1e-9`, `atol=1e-12`, `equal_nan` handled by the row classes above rather
  than by the comparison.
- **Per-column overrides**, because a single absolute tolerance across metres (0–105), m²
  (0–7140) and probabilities (0–1) violates the repo's own rule that you measure at the
  scale you assert at.
- An override **requires a written rationale**, on the same footing as an adjudication,
  and changing one is a change to the lock.

Tolerance applies to **float columns only**. Emitted columns also include counts and
booleans, where the comparison is **exact equality** — a tolerance on an integer count is
meaningless and would silently absorb an off-by-one.

**A cross-leg type mismatch is its own loud failure**, never an implicit cast inside the
comparison. This is not hypothetical here: the `int64` vs `Int64` vs `object` id trap is
already flagged for `player_id` under ADR-019, and it is exactly the shape where a real
defect reads as `identical` because the comparison quietly reconciled the two sides.

The check compares the **kind of the actual values** — numeric / boolean / other, inferred
from the non-null entries — and **not the declared dtype**, before any value comparison runs.
That distinction is load-bearing rather than pedantic. pandas cannot hold NaN in `int64`, so
an integer column that declines on some Leg A rows upcasts to `float64` while Leg B stays
`int64`. A declared-dtype comparison fires on that and **aborts the audit on `partial_nan`** —
the observation this spec calls the expected outcome on the visibility axis. Inferring the
kind from values subsumes the all-NaN case, the partial-NaN case and the real ADR-019 trap
under one rule, instead of accruing an exemption per case; a leg with no values makes no
claim and is skipped.

### Adjudication — human-written, rationale mandatory

| Adjudication | Admissible from | Rationale required |
|---|---|---|
| `works` | `identical` | only if tolerance is non-default (see below) |
| `silent_degrade` | `differs`, `partial_nan` | **yes** |
| `differs_by_design` | `differs`, `partial_nan` | **yes** |
| `honest_nan` | `all_nan`, `partial_nan` | **yes** for `partial_nan` |
| `not_exercised` | `no_signal` | **yes** |
| `raises` | `raises_a` | no |

`partial_nan` is admissible to three adjudications precisely because it is ambiguous — a
column NaN on the actions where the keeper was invisible and finite elsewhere could be any
of them, and only a human reading the feature can say which. That is why it carries a
mandatory rationale in every case.

**The manufactured-`works` path is closed here.** The tolerance section above notes that
loosening a tolerance converts `differs` into `identical`. But `works` from `identical`
would otherwise require no rationale, so a loosened tolerance yields a **rationale-free
`works`** — and the tolerance override's own rationale lives in a different field, reviewed
at a different time. Two mitigations that only work together:

- the column's tolerance is surfaced in the registry entry **adjacent to the adjudication**,
  not in a separate table
- `works` requires a rationale **whenever that tolerance is non-default**

Naming a risk in a spec does not close it; this is what closes it.

The CI gate asserts the **observation** against the registry, never the adjudication.
Repair `add_gk_influence` and the observation changes, CI fails, and the adjudication is
forced to be revisited. That is the lock — without pretending a machine can adjudicate.

`differs_by_design` is not decorative. Pitch control at zero velocity is a well-defined
*positional* model — weaker, but not fabricated. Distinguishing that from `_gk_influence`'s
silent zero-fill is a judgement, and the written rationale is the reviewable artifact.

### Structural impossibility — an orthogonal annotation, not a verdict

`structurally_impossible: <named reason>` is a separate optional field that **must
co-occur with an `all_nan` or `raises_a` observation**. A function annotated structurally
impossible that observes `identical` is a contradiction the gate catches. This keeps the
useful distinction — `add_actor_pre_window` can *never* work on one-frame-per-action,
versus *this happened to return NaN here* — while making the claim falsifiable rather than
declared.

### State-vocabulary completeness gate

Four consecutive revisions of this spec acquired the **same defect**, each time introduced
while repairing the previous one:

| Rev | Defect |
|---|---|
| 1 | Registry locked keys, not content |
| 2 | `infeasible` defined but unproducible by the decision rule |
| 2 | `partial_nan` an input the column rule could not classify |
| 3 | Visible-area qualifier locked but derived from a CI-excluded layer |
| 3 | Applicability class asserted re-derivable with no derivation given |
| 4 | `row_nan_b` and `raises_b` missing from the row table |
| 4 | Tightening `identical` orphaned the sparse-domain column |
| 4 | `raises_b` defined at Level 1, absent from the precedence table |

Every one has one shape: **a state is introduced at one level and not propagated into every
table that claims completeness.** Reviewing again would find the next instance; it would not
stop the next instance. A mechanical check does.

The design vocabulary is declared **once**, as the single source, and is **namespaced** —
`call_outcome.*`, `row_class.*`, `observation.*`, `adjudication.*`, `kind.*`,
`applicability.*`. Namespacing is not tidiness: rev 4 had an observation named `raises` and
an adjudication named `raises` denoting different things, while the parallel `raises_b`
appeared in both vocabularies denoting the same thing. A flat name set would have conflated
the first pair and been unable to express that the second is deliberate.

Two names are shared across `call_outcome` and `observation` — `raises_a` and `raises_b` —
because those states pass straight through unchanged. That is declared, not incidental.

A test then asserts:

- every `call_outcome` either appears as a precedence rule **or** is the declared
  precondition for row classification (`both_succeeded` is the only one of the latter kind)
- every `row_class` is consumed by at least one precedence rule
- every `observation` the precedence table can produce carries a `kind`
- every `observation` with kind `adjudicated` or `budgeted` appears in the adjudication
  admissibility table
- every `observation` with kind `terminal fixture failure` is **absent** from it
- every `adjudication` is reachable from at least one `observation`
- every `applicability` class is producible by the probe procedure

This is a small test over the registry schema itself, and it is the same both-directions
discipline the spec already applies to `tracking.__all__` — turned on the design's own
vocabulary. It is what converts this from *reviewed four times* into *cannot silently acquire
a fifth hole*.

### Granularity

Verdicts are recorded **per emitted column**, not per function. `add_action_context`
already splits — its three positional columns against `actor_speed`, which observes
`all_nan` on Leg A (see below) — and a function-level verdict would erase exactly the
distinction that matters.

Each registry entry carries two independent observation/adjudication pairs — **velocity
axis** and **visibility axis** — plus the optional structural annotation, the per-column
tolerance, and the visible-area applicability class.

## Layer A — synthetic behaviour matrix

### Leg construction

**Leg A is built by calling `snapshot_to_tracking_frames` on synthetic 360-shaped input**,
not hand-assembled: the fixture cannot drift when the producer changes, and the audit
exercises the code path the NWSL data will actually hit.

Dtype trap handled explicitly: canonical `player_id` is `int64` while provider variants are
`object` or `Int64` — the reason ADR-019's `ids_match` exists. A hand-built fixture
silently picks one and can mask a real dtype defect, so the synthetic input is
parameterized over id dtype.

**Leg B** is a full velocity-bearing frame set with real `vx`/`vy` and a temporal
neighbourhood, constructed so **positions at each linked frame are identical to Leg A's**.
Leg B necessarily has more rows; the invariant is per-linked-frame position equality.

### The two axes vary independently

Load-bearing, and an earlier draft got it wrong in exactly the way Layer B's 2×2 exists to
avoid:

- **Velocity axis** — roster held **fixed**; only kinematics vary.
- **Visibility axis** — velocity held **fixed**; roster varies: GK absent / outfield
  defender absent / full complement (control).

Varying both together makes every verdict unattributable.

### Motion model

Leg B's trajectory is **non-degenerate by construction** — varying speed *and* direction
across the neighbourhood. Constant-velocity motion makes every acceleration-dependent
quantity identically zero in both legs and reads a false `works`. The trajectory is
specified in the fixture and stated in the report, because "what motion did the fixture
have" is the first question any reader should ask of a `works` verdict.

### Neighbourhood length is a hard fixture requirement, not a happenstance

Leg B's temporal neighbourhood must be **at least as long as the longest window among the
enumerated features**, and the fixture asserts it rather than happening to satisfy it.

The requirement is load-bearing because of how it fails. `structurally_impossible` must
co-occur with `all_nan` or `raises_a`. `add_actor_pre_window` is this spec's own example of a
structurally impossible feature — it needs frames *before* the action, which Leg A cannot
have. But if Leg B's neighbourhood is shorter than that feature's window, **Leg B is NaN
too**: the rows classify `row_nan_both`, the column observes `no_signal`, and it is forced
to `not_exercised` — at which point the structural annotation is inadmissible and the
distinction it exists to preserve is silently lost. A fixture that is merely *usually* long
enough would lose it intermittently, which is worse.

### Non-vacuity — distinguishability, not `differs`

A global "at least one column differs" gate is nearly vacuous over 33 functions: 32 dead
columns plus one live difference passes it.

But a **`differs` canary cannot be pre-registered without pre-judging the audit.** Naming a
column that silently degrades in advance is asserting the finding the spec insists must be
established by execution. An earlier draft named `add_action_context`'s `actor_speed` and
was provably wrong: `_snapshot.py:122` sets `speed=np.nan` on every snapshot row, and
`_kernels.py:80,88` initialises `actor_speed` to all-NaN and fills only where
`pd.notna(speed)`. On Leg A that column is all-NaN — observation `all_nan`. It is not the
canary; it is the **model citizen**, the one velocity consumer that already degrades
honestly.

The assertion's actual job is to prove the legs are not secretly equivalent. So:

- **Canary:** assert `actor_speed` observes **anything other than `identical`**. All-NaN on
  Leg A against finite on Leg B proves distinguishability without predicting a defect.
- **Set equality:** assert the full observation map equals the pre-registered map — the same
  registry lock described above, not a second mechanism.

### Fixture is versioned state

Locking observations pins the fixture as well as the library: any trajectory, roster or
dtype change flips observations and fails CI identically to a real regression. That is
acceptable, but the failure must be readable. The fixture carries a version, and the
assertion message includes it, so **"the library changed" and "the fixture changed" are
distinguishable at the point of failure** rather than after a debugging cycle.

### Enumeration boundary

- All 33 `tracking.__all__` `add_*` exports. `add_gradientsports_player_ids` is a jersey
  helper; its expected observation is **recorded as a prediction that fails loudly if
  violated**, not narrated as an exception.
- `gkdv` and `xtgk` v2 expose no `add_*`; enumerated by public compute entry point
  (`build_ghost_frames`, both arms, `compute_xt_gk_v2`).
- `spadl.add_restart_coordinates` accepts `frames=` and is frame-consuming despite living
  outside `tracking`. Enumerated as a boundary case.

## Layer B — real-data coverage

### Sample design

Verified against `data/competitions.json` in the StatsBomb open-data repository (fetched
2026-08-04): twelve competition-seasons carry 360, of which the women's ones are the Women's
World Cup 2023 and UEFA Women's Euro 2022 and 2025 — **all tournaments**. There is no women's
league with 360, so sex and production tier are confounded and a naive pair varies both axes
at once.

That count and that "all tournaments" claim are prose verification, and this spec's own rule
is that prose rots. Wherever either appears in the **artifact**, the driver derives it from
`competitions.json` at run time rather than restating it. The three IDs actually sampled are
resolved and name-asserted (below), which is the part that changes what gets measured.

| | Tournament | League |
|---|---|---|
| **Men's** | FIFA World Cup 2022 (43/106) | MLS 2023 (44/107) |
| **Women's** | Women's World Cup 2023 (72/107) | *(none exists in open data)* |

- WWC2023 vs WC2022 — isolates sex at fixed production tier.
- MLS2023 vs WC2022 — isolates production tier at fixed sex.

**5–10 matches per cell**, not one. Match-level production variance (stadium, camera rig,
weather, fixture importance) is plausibly as large as the between-competition effect, so
n=1 per cell would make each contrast an anecdote. Match count is a driver flag and the
pass is sharded, so it resumes.

> **Amended 2026-08-04 after measurement: the cell CONTRAST is withdrawn. The aggregate and its
> DISPERSION are the deliverable.**
>
> **The unit of analysis is the MATCH, not the event.** Goal kicks within a match share one
> broadcast, one camera rig, one production crew — they are clustered. Eight matches gives ~104
> goal kicks per cell, which *looks* like a ±8pp proportion, but the effective n is ~3. This
> repo already encodes that reasoning (`causal/power.py`: `icc_power_curve`, whole-cluster
> `placebo_shift`), and applying it here says a credible cell contrast needs ~15–20 matches per
> cell — 45–60 matches of network pull, to support a comparison that was always a *construction*
> to dodge a confound the open data makes unavoidable anyway (no women's league carries 360).
>
> So: keep the default at 8, and report **per-action-type frame existence with match-level
> dispersion** — median, IQR, range, `n_matches` — instead of a per-cell contrast. The dispersion
> IS the finding: the measured range was 0–50% across 9 matches, and a club planning around
> goal-kick coverage needs that, not a point estimate of 23%. The three cells stay in the sample
> for breadth of production conditions; they simply stop carrying a causal claim.
>
> The pass is sharded and additive, so this is not one-shot: run 8, read the dispersion, extend
> if it warrants.

**IDs are resolved and asserted, not trusted.** The driver resolves each competition/season
ID against `competitions.json` and asserts the resolved competition and season *names*
match expectation, so ID drift in the open-data repo fails loudly instead of silently
sampling the wrong tournament. Prose verification does not survive an upstream renumber.

**NWSL relevance, stated without collapsing the axes:** MLS 2023 bounds the **tier** axis,
WWC 2023 bounds the **sex** axis, and **NWSL sits at a combination the open data has no
observation for.** Calling a men's league "the closest analogue" for a women's league would
re-import the confound the 2×2 exists to separate. The honest statement is a bounded
region, not a point — and a stronger opening for the commercial conversation than a single
number.

**NWSL itself is not in the 360 list.** The open data cannot answer the NWSL question
directly.

### Reported metrics

- **frame-existence rate** per SPADL action type — how many actions of that type EXIST and how
  many received a freeze-frame at all, counted from the ACTION side
- **two** keeper-visibility rates per action type, with counts alongside every rate
- players-per-freeze-frame distribution and implied missing count
- measured visible-pitch fraction per action type, from `visible_area`, normalised by
  StatsBomb's **120×80** pitch (not SPADL's 105×68, which yields ~1.34 for a full frame)

> **Amended 2026-08-04 after measurement.** This list originally named a single metric — "share
> of GK-domain events where the *defending* keeper is visible" — and both halves of that were
> wrong.
>
> **WHICH keeper is "the" keeper depends on the action type.** On a goal kick or a save the
> keeper IS the actor, so `keeper AND NOT teammate` excludes them BY CONSTRUCTION. Measured:
> `goalkick` and `keeper_save` read exactly 0.000, which reported as-is would have told a club
> its goal-kick coverage was nil. Both `defending_gk_visible_rate` (shots, crosses) and
> `acting_side_gk_visible_rate` (distribution, saves) are emitted; the consumer picks by type.
>
> **A per-frame metric is structurally blind to an action that got no frame.** For goal kicks
> that is the entire story — 12 in the sampled match, 1 with a frame. Frame existence is
> therefore counted from the action side and listed first, because it bounds everything below
> it: a keeper-visibility rate over 8% of the domain is not a coverage number.

### Artifact provenance

Layer B is deselected from CI but its artifact will be quoted in a commercial conversation
months from now, so it stamps and renders its own staleness: generation date, competition
and season IDs *and resolved names*, the full match-ID list, `statsbombpy` version, plus the
ADR-037 `run_commit` / `run_tree_dirty` from `require_clean_tree`. Code provenance and data
provenance are both required; neither substitutes for the other.

### Implementation constraints

- `statsbombpy`, `importorskip`-guarded, network-gated `@pytest.mark.e2e` — mirrors
  `tests/test_xthreat_statsbomb_e2e.py`. Not a committed extra; self-skips without it.
- Corpus pass: adopts `scripts/_driver.py` (`for_each`, per-match shards, `reconcile`) per
  ADR-052, and `require_clean_tree` in `main()` per ADR-037. Both CI-gated already.
- Competition/season IDs and match count are **parameters**, not constants.

## Layer B′ — the inverse question (`visible_area`)

The audit as originally scoped asked only "which of our 33 aggregators survive freeze-frame
input." That is half of "have we utilised SB360 as much as possible"; it never asked what
SB360 carries that we have **no metric for at all**.

The concrete instance is the per-event **`visible_area` polygon**. Verified: zero handling
anywhere in `silly_kicks/` — the only match in the package is the event-type strings
`"Camera On"`/`"Camera off"` at `spadl/statsbomb.py:64-65`.

It matters *to this audit*, not merely alongside it. The visibility axis otherwise treats
off-camera players as an unquantifiable fabrication source. The polygon bounds it: for a
feature querying a **region** — defenders in the triangle to goal, receiver zone density,
cover-shadow lanes, a pitch-control grid — you can measure what fraction of that query
support was inside the camera's view, converting "0 defenders in the triangle" from
ambiguous into either *0 defenders, region fully observed* or *0 defenders, region 40%
observed*.

### Applicability class — three categories, not two

| Class | Meaning | What is recorded |
|---|---|---|
| `region_support` | Fixed spatial query region | Observed fraction of that region |
| `no_support` | Scalar off the actor's own row (e.g. `actor_speed`) | `not_applicable`, recorded explicitly |
| `support_data_defined` | Query support is defined *by the visible players themselves* | Roster completeness, **not** a coverage fraction |

`support_data_defined` is the category most likely to be quietly wrong, and it needs naming
for that reason. Team shape's convex hull, inter-line gaps and the Delaunay role grid all
define their support from whoever is visible — so the support is 100% "observed" **by
construction**, precisely in the case where invisibility does the most damage. A coverage
fraction there is circular and would read as reassurance. What is honest instead is roster
completeness: visible players against the expected complement, per team.

`no_support` is an explicit recorded value, never a blank — an empty field cannot be
distinguished from an unfilled one, which is how a coverage denominator becomes a tactical
signal (ADR-042).

### Deriving the class — two perturbation probes, in order

An earlier draft asserted the class was "derivable in Layer A" without saying how. If the
answer had been *a human picks one of three*, that would put a declaration inside the
**locked** half of the registry — the exact observation/adjudication conflation this design
exists to prevent, in its third location.

It is genuinely derivable, by perturbing player positions at fixed roster and fixed polygon
— so neither probe depends on polygon masking, which would collapse into roster variation
and discriminate nothing.

**Probe 1 — extreme-player displacement.** Move a player positioned far from the action,
beyond any plausible query radius, without adding or removing anyone.

- Output moves → the feature's support is defined by the player set → `support_data_defined`
- Output unchanged → not data-defined; continue to probe 2

The probe works because a convex hull, an inter-line gap or a Delaunay role grid is
*defined by* its extreme members, whereas a triangle-to-goal or a receiver zone is defined
by action geometry and is indifferent to a player who was never inside it.

**Probe 2 — near-player displacement.** Move a player close to the action.

- Output moves → the feature has spatial query support → `region_support`
- Output unchanged → the feature reads only the actor's own row → `no_support`

**Order is load-bearing, and a feature can satisfy both.** Probe 1 runs first and wins:
data-defined support is the dangerous property, because it is the one where the coverage
fraction reads as reassurance while being circular.

Each probe's measured deltas are recorded, so a class assignment can be checked against the
numbers that produced it rather than taken on trust.

### What is locked and what is stamped

The applicability class is a **code property**, derived by the two probes above, so it lives
in the registry and CI re-derives it on every leg.

The **measured coverage fraction** is a **data property** of specific broadcasts. It exists
only in Layer B, which is network-gated and deselected, so CI can never re-derive it. It
therefore lives in the Layer B report, stamped and dated — **not in the registry.** Putting
it in the registry would create a locked field CI cannot check, which is round one's finding
relocated into this section.

### How the polygon reaches the harness

Via a **side table keyed by `action_id`, inside the audit harness.** The `snapshots`
contract (`action_id`/`team_id`/`is_goalkeeper`/`x`/`y`) is **not** extended, and
`snapshot_to_tracking_frames` is **not** modified, this cycle. Stating the seam explicitly
closes the route by which scope would otherwise creep into a public contract.

**Scope boundary held:** the audit consumes the polygon in its own harness and report.
Whether the library should gain a public seam is a **finding**, not part of this cycle —
consistent with "the audit reports; it does not fix."

## Limitations

1. **GK-event n.** Even at 5–10 matches per cell, GK-domain events are ~10–25 goal kicks
   and a handful of saves per match. Rates are reported with counts alongside, never as
   bare percentages.
2. **Empty cell.** With no women's league carrying 360, the 2×2 has no fourth cell. Each
   contrast is clean; no interaction can be estimated.
3. **Broadcast ≠ delivered data.** Open-data 360 reflects those specific broadcasts. An
   NWSL commercial feed may differ in either direction.
4. **The audit reports; it does not fix.** `silent_degrade` adjudications are findings.

## Scope boundary

**In:** the paired synthetic fixture and behaviour matrix; the real-data coverage driver and
report; `visible_area` consumed audit-side; the observation/adjudication registry and its CI
lock; a `docs/research/sb360_coverage/` artifact.

**Out, surfaced for scoping rather than decided:**

- a `providers/statsbomb/` freeze-frame parse port making SB360 a first-class producer
- a public library seam for `visible_area`, if the audit shows it earns one
- fixing whichever `silent_degrade` adjudications turn up (`add_gk_influence`'s zero-fill is
  the strong candidate)
- extending `velocity_unavailable_by_design` to the remaining velocity-touching modules that
  ignore it. The exact count is deliberately not asserted: 31 files under `tracking/` mention
  `vx`, but that set mixes genuine consumers with schema, preprocessing and reflection
  infrastructure, and separating them is a job for the audit rather than a grep

## Sequencing constraint

`scripts/_provenance.py` counts **untracked** files as dirty, so this spec and the plan must
land in the **first commit** of the branch. An uncommitted doc makes every artifact driver
`SystemExit` before it does any work.

## Testing

- Registry key meta-assertions, both directions, against `tracking.__all__`.
- **Row-class exhaustiveness:** assert every (row, column) pair matched **exactly one** row
  class, with the unmatched count and a sample of unmatched pairs in the failure message. The
  claim was prose in an earlier draft and was false; prose is what this audit replaces.
- **Per-column liveness:** every column shows at least one finite value in at least one leg,
  per axis. A column failing this is `no_signal` → `not_exercised`, and the **count** of
  `not_exercised` entries is pre-registered and locked so it cannot quietly grow.
- **Registry observation lock:** Layer A re-derives every observation and asserts equality
  with the registry. Failure messages carry the fixture version, so a fixture change is
  distinguishable from a library regression at the point of failure.
- **Adjudication well-formedness:** every adjudication admissible from its observation;
  rationale present wherever the table requires it — including `works` on any column with a
  non-default tolerance; every `structurally_impossible` co-occurring with `all_nan` or
  `raises_a`; every tolerance override carrying its own rationale.
- **Non-vacuity:** `actor_speed` observes anything other than `identical`, plus observation-map
  set equality. Measured values pasted into assertion messages. Note this is now the *weakest*
  of the vacuity guards — per-column liveness is the one carrying the load.
- **Applicability class** re-derived by the two perturbation probes and asserted against the
  registry, with each probe's measured delta in the failure message.
- **Aggregation exhaustiveness:** assert every column matched exactly one precedence rule,
  with the unmatched columns named in the failure message. Rules 5–8 are argued exhaustive in
  prose above; the assertion is what makes that true rather than believed.
- **State-vocabulary completeness:** the seven invariants above, over the single declared
  vocabulary. This is the gate that stops the recurrence rather than patching its latest
  instance, and it must be landed RED like the other registry gates.
- **Fixture-integrity failures** — `raises_b` and `leg_b_declined` — never reach the registry.
  `leg_b_declined`'s message carries the `row_nan_b` count and a sample, so a single row is
  distinguishable from systematic decline.
- **Fixture preconditions:** Leg B's neighbourhood is at least the longest enumerated feature
  window; dtypes match across legs per column. Both assert before any comparison runs.
- **Non-float comparison:** count and boolean columns compare exactly; a tolerance entry for a
  non-float column is itself a failure.
- Layer B driver: conservation over its own keys plus `_require_injective`, per ADR-052; and
  resolved competition/season names asserted against expectation.
- Layer B is `@pytest.mark.e2e` and deselected; the registry gate and synthetic matrix run on
  every leg.

## Not in scope

- Any change to library behaviour, aggregator surface, or default xfn lists.
- C4 regeneration — no new action-coupled aggregator; documented count unchanged.
- Model retraining — nothing here moves a feature value.
