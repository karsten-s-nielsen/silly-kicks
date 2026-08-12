# ADR-055: Goal-map unification and the observed-region seam

| Field | Value |
|---|---|
| **Date** | 2026-08-08 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

The rule "which goal does this team defend" was implemented **ten times** across five modules,
each spelling a variant of `0.0 if same_id(team, home_team_id) else 105.0`. Three problems
followed from that, and only the third had ever been written down.

**It is identity-keyed, not direction-keyed.** `home_team_id` says who the home team is; it says
nothing about which way anyone attacks. The equivalence holds only while frames are in the
`convert_to_frames` home-attacks-right convention, which is a property of the caller's pipeline
rather than of the function's inputs. This is ADR-051's D3 class, and `add_gk_influence` /
`add_cover_shadows` were its last two live members among the aggregators (measured movement when
`home_team_id` was varied on FIXED frames: `gk_pitch_control_share_weighted` 0.108532,
`gk_closing_time_min_s__six_yard_box` 4.38062 s, `blocking_score` 148.83).

**It has no period term.** A team swaps ends at half time. Every one of the ten sites answered the
same for period 1 and period 2.

**It fails OPEN.** Each site had a total function: some end was always returned. On frames where
the end is genuinely unresolvable — no keeper, an NA team identity, all-NaN coordinates — the
sites returned a confident 105.0 (`nan < 52.5` is False), and the resulting number was
indistinguishable downstream from a measured one.

Separately, StatsBomb 360 ships a `visible_area` polygon per event, and every count feature in
the library treats "nobody there" and "nobody VISIBLE there" as the same observation.

## Decision

One seam, `resolve_defended_goals(frames) -> GoalMap`, built **once per match from the full
frames** and **threaded** into per-frame functions where it *replaces* `home_team_id`; consumers
call `get` (own end) or `attacked_goal` (opponent's end), never a plain dict. An eleventh
implementation fails CI. Plus `tracking/_visibility.py`, shipping `point_observed`,
`region_observed_fraction` and `add_visible_area_coverage` so a consumer can distinguish an
absence from a non-observation.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Keep `home_team_id`, fix each site in place | No API change; smallest diff | Ten copies remain, so an eleventh is still free; the period term has to be added ten times | Fork-by-duplication is the defect, not its symptom |
| B. Build the map per frame inside each function | No threading; no signature change | A different estimator, and unresolvable for a third of team-frames on sparse providers (see *The per-frame cost* below) | A default that re-admits it at every call site that forgets to pass one |
| C. Deprecate `defended_goal_x`, keep it as an alias | No import breaks | A consumer keeps a name whose value changed | A hard rename fails loud at import; an alias fails silently |
| D. **Chosen**: one seam, threaded, `home_team_id` removed | One implementation; period-scoped; refuses rather than guessing | 15 breaking public signature changes across two packages | — |

Two sub-decisions were taken by the owner during implementation, both after the alternatives had
been measured rather than argued:

* **`select_back_line_players` takes `defends_x0: bool`,** not `goal_x: float`. Both remove the
  identity key; only the bool avoids a cascade. `_packing.py` would have had to express a goal end
  as `0.0 if same_id(...) else 105.0` to supply a float, which is the very fork the population gate
  rejects, so `compute_packing_metrics` would have needed a `GoalMap` too — three more breaking
  changes and ~26 files, in a family this cycle's own scope boundary leaves to ADR-051.
* **The four public GK scalar helpers drop `home_team_id`** rather than keeping it unread. The
  repo has both precedents (`space_creation` retains an unread one, "D3 retired by disuse"), and
  the cycle's rule — a dead required parameter is the shape being deleted — was applied
  consistently rather than half.

## Consequences

### Positive

* The goal-end rule exists in exactly ONE place, pinned by a semantic AST gate
  (`tests/tracking/test_goal_map_population.py`) that matches `IfExp`, `if`/`else`-assigning-one-name
  and `np.where` — landed RED and observed failing on all 10 forks across 5 modules.
* Every site gains a period term it structurally lacked.
* Unresolvable ends REFUSE (`GoalEndUnresolvedError`, a `ValueError` subclass) instead of
  returning 105.0. Policy lives at the edge: per-frame functions raise, the `add_*` aggregator
  catches by name and emits a NaN row. Pre-checking at the aggregator was rejected because it
  duplicates the exact lookup the callee is about to do — a second implementation of the decision,
  inside the commit that deletes second implementations.
* `attacked_goal` is a real lookup of the opponent's entry, never `105.0 - get(...)`: the
  arithmetic identity is wrong on a degenerate map, where it would say a team attacks the goal it
  defends.

### Negative / risks

* **15 breaking public signature changes**, across `tracking` and `gkdv`, plus two renames
  (`defended_goal_x` deleted; `visible_fraction` → `observed_pitch_fraction`). All fail loud at
  import or call. Precisely: 14 of the 15 are in an `__all__`; the fifteenth,
  `compute_gk_influence`, is public by documentation rather than by export — CLAUDE.md's TF-15
  row lists it as the family's public surface — so it is counted, and the distinction is
  recorded rather than rounded away.
* **VAEP/tracking retrain trigger.** `add_gk_influence` and `add_cover_shadows` change value
  wherever identity and direction disagreed, and `add_cover_shadows` is now **keeper-dependent on
  freeze-frame input** — on SB360's `gk_absent` roster its five columns collapse from `all_nan` to
  `no_signal`, because the outfield fallback guesses both teams at the same end (measured means
  56.9 and 76.5, both past the 52.5 midline) and `attacked_goal` refuses a degenerate map. That is
  a correction: previously both legs produced numbers the frames could not support.

## The per-frame cost, measured — and a spec figure that does not reproduce

The spec's §15 headline was "78.8% wrong". **It does not reproduce on the four committed slim
fixtures**, and it is cited nowhere in this repo as a result. Measured here instead, comparing a
map built per FRAME against the per-MATCH map, over every `(game, period, team)` the per-match map
resolves:

| provider | team-frames | per-frame map DISAGREES | `attacked_goal` unresolvable |
|---|---|---|---|
| skillcorner | 504 | 36 (**7.1%**) | 180 (**35.7%**) |
| metrica | 1,112 | 24 (2.2%) | 686 (61.7%) |
| gradientsports | 3,600 | 0 (0.0%) | 0 (0.0%) |
| sportec | 1,104 | 0 (0.0%) | 0 (0.0%) |

Two things follow, and both are worth stating plainly rather than rounding into the spec's number.

**The cost is PROVIDER-DEPENDENT, not universal.** On dense tracking (sportec, gradientsports:
every keeper in every frame) a per-frame map is exactly right, and building once per match costs
nothing. The damage is concentrated in sparse broadcast detection — SkillCorner detects a keeper in
~19.6% of frames (ADR-038) — which is precisely the provider class this seam exists to serve.

**The spec's second number DOES reproduce.** Its 34.2% unresolvable rate lands on SkillCorner's
measured 35.7%; only the 78.8% does not, against a measured maximum of 7.1%. The design decision
rests on the reproduced half: a third of team-frames losing direction entirely is decisive on its
own, and the old sites answered those frames with a confident 105.0.

Recorded because shipping the unreproduced figure in the ADR, `CLAUDE.md` and the CHANGELOG would
have put a number nobody could re-derive at the centre of the cycle's rationale — the shape this
repo already names as a plausible result from a computation that did not happen.

## Gate C, and why Gate B was retired for two aggregators

The sharpest review finding of the cycle: re-keying off `home_team_id` makes **Gate B vacuous**,
so the cycle would have deleted the only detector for the class it fixes. Gate B varies
`home_team_id` on fixed frames; once the parameter is gone it skips on `role="unused"`, and the
AST gate cannot see a direction *bool*.

**Gate C** (`test_gate_c_goal_map_is_the_direction_source`) is the same question one variable
further out: hold the frames fixed, swap the MAP, require the declared columns to MOVE. It
reproduces the recorded D3 magnitudes exactly — `share 0.108532`, `closing_min 4.38062`,
`closing_mean 4.02205`, `blocking_score 148.83` — which is the evidence that it detects the class
Gate B used to.

Two honest qualifications, both recorded at the gate:

* **Gate C proves the map is CONSULTED, not that the right accessor was chosen.** `get` and
  `attacked_goal` both move when the map is swapped. The correctness half is
  `tests/tracking/test_goal_map_consumers.py`.
* **Gate C is structurally blind to `_closing_time_per_series`.** The plan asserted that a
  one-column result would mean that path was missed. Executing the defect — patching it back onto
  a self-built map and running Gate C on `add_gk_influence` — leaves Gate C **GREEN**, because
  `add_gk_influence` never calls it; its closing-time columns come from `_gk_influence_at_actions`.
  That path is covered instead by `test_closing_time_helpers_read_the_injected_goal_map`, verified
  red against the same executed defect.

## The pre-registered measurements: M2 and M3

The spec fixed three measurement rules BEFORE measuring, so an answer could not be chosen after
seeing it. **M1 was run on the DGX owner corpus in 4.77.1 and is ZERO** (see below). M2 and M3 are
measurable on the committed slim fixtures and were run:

| provider | M2 — GK-less `(game, period, team)` groups | M3 — end DIFFERS between p1 and p2 |
|---|---|---|
| sportec | 0 of 4 | 0 of 2 |
| gradientsports | 0 of 4 | 0 of 2 |
| skillcorner | **2 of 4** | **1 of 2** |
| metrica | **3 of 4** | **1 of 2** |

**M2 is non-zero**, so the `allow_guess=True` decisions threaded through this cycle are
LOAD-BEARING rather than vacuous: on metrica three of four `(game, period, team)` groups carry no
keeper rows at all, and without the outfield rung those groups would resolve to nothing.

**M3 is non-zero, and it is the finding that matters.** The spec's pre-registered reading:
*non-zero → the three `features.py` sites and `_gk_influence.py:318` are producing wrong values
today for every second-half action on the affected providers.* Measured:
`('skillcorner_1899585', 'away_team')` defends x=0 in period 1 and x=105 in period 2;
`('Sample_Game_3', 'home_team')` the same in reverse.

**It survives `play_left_to_right`.** Re-measured on oriented frames, both flips persist
unchanged — so this is not "the caller forgot to orient", which ADR-029 already covers. The
period term is therefore not a theoretical nicety: on two of four committed providers the
identity-keyed sites, which had no period term at all, answered second-half actions with the
first half's end.

That converts §2.7's "no value effect" into a value change on those providers, which is why the
retrain trigger in this ADR is stated unconditionally rather than hedged. It also means the
sportec/gradientsports byte-identity is a property of THOSE fixtures, not a general claim.

### M1, measured on the full owner corpus (4.77.1): ZERO

Pre-registered rule: *zero → no retrain, record the count; non-zero → retrain trigger, handled as
its own weights cycle.* Measured on the 179-match DGX pining corpus, conservation asserted (every
expected match produced exactly one shard, zero failures):

| provider | matches | GK rows | NA-team GK rows |
|---|---|---|---|
| gradientsports | 64 / 64 | 23,727,139 | **0** |
| idsse | 7 / 7 | 2,005,288 | **0** |
| skillcorner | 108 / 108 | 9,602,782 | **0** |
| **total** | **179 / 179** | **35,335,209** | **0** |

**So the bundled `GhostGkModel` weights are NOT contaminated, and no retrain is triggered by this
ADR.** The retired fallback (`0.0 if same_id(gk_team, home_team_id) else 105.0`, which returns a
constant 105.0 for any NA-team keeper because `same_id(NA, home)` is False) was reachable in code
and never reached by this corpus.

Two properties make the zero load-bearing rather than incidental. The count is a **superset** of
what enters training — the extractor further restricts by domain, subsample and link filter — so
zero over the superset bounds the training set at zero. And the selection is **identical** to the
extractor's: the probe's `fillna(False)` differs from the extractor's raw `.astype(bool)` only if
`is_goalkeeper` can hold NA, which it cannot here, because that raw cast RAISES on NA in both
`boolean` and `object` dtypes and the model trained successfully on this corpus.

The honest reading, and the same one the coverage re-run produced: **this cycle's repair closes a
path that is real in the code and absent from the corpus.** That is worth keeping — the guard costs
nothing and the next corpus may differ — but it is not a repair of a wrong published number, and
recording it as one would be a fabrication in the direction that flatters the change.

## The D3 unit pin moved

`test_defensive_line_d3_unit_is_enumerated` pinned `{_defensive_line, _packing, _gk_influence}` as
the files reading `home_team_id`. `_gk_influence` left the set, so the pin is now the other two,
with the reason recorded and the departure asserted separately — a member silently dropping out is
exactly how a partial re-key would hide.

## The `_snapshot` dtype pin was DROPPED, and the spec claim corrected

The design (§2.6) recorded that `_snapshot.py`'s `pd.concat` yields `Int64` on pandas 2.3.3 and
`Float64` on 3.0.3, and prescribed casting to `TRACKING_FRAMES_COLUMNS`. Measured on the pinned
resolver (pandas 2.3.3):

* the concat yields **`float64`**, not `Int64`;
* the prescribed cast is **unimplementable** — the schema declares `int64` for `player_id` and
  `team_id`, the ball row is NA in both, and `int64` cannot hold NA (`IntCastingNaNError`), so it
  raises on every snapshot; the declaration is also not what the library's own adapters emit
  (`object` ids for the DFL providers), so forcing it would reject string ids;
* a `restore_id_dtype`-based pin changes **nothing** for numpy-int, nullable-`Int64` or object id
  sources, and with the pin excised **0 of 2** tests written for it went red.

Shipping a change nobody in CI can observe, justified by a measurement that does not reproduce, is
the "plausible result from a computation that did not happen" shape this repo already names. The
residual concern — a frame builder whose id dtype moves with pandas 3 — is real and is a recorded
follow-up, not a silent no-op in this commit.

## The observed-region seam

`point_observed` returns `bool | None`, because `False` is a CLAIM the camera did not see a point
and a missing polygon supports no claim. `region_observed_fraction` takes an `(M, 2)` POLYGON, not
a bounding box: a bbox can only OVER-report coverage for a triangle, i.e. fabricate observation.
`visible_area_fraction` is NaN for every non-`observed` token — never 1.0, never 0.0.

Sutherland–Hodgman requires a convex CLIP, so the **polygon is always the SUBJECT** and the
**region is the clip**: that puts the convexity requirement on the argument the caller constructs
(a triangle, a zone, a pitch-control cell) rather than on broadcast data nobody controls, and a
concave region is REFUSED rather than silently mis-clipped. The primitives live in a neutral
`silly_kicks/_polygon.py` — the `id_compat` / `reflection` position — because `providers/` has no
runtime dependency on `tracking/` and adding one would invert the port layering.

Wiring coverage INTO the count features is deliberately NOT in this cycle: it changes existing
values and decides for the consumer what a partial observation means (ADR-009).

### The vocabulary was defined and then not used by its own producer (4.77.1)

The token split above is the seam's central claim, and the producer feeding it violated it.
`shape_snapshots` dropped any polygon that converted to an empty array, so a published-but-2-vertex
`visible_area` produced no row at all and read downstream as `no_polygon` — "nothing published",
when the truth was "published and unusable". That is precisely the collapse `degenerate_polygon`
exists to prevent, committed in the module that defines both tokens.

4.77.0 recorded the adjacent `polygon_to_spadl` crash (`reshape(-1, 2)` raises `ValueError` on an
odd-length flat list, mid-corpus) but declined to fix it, on the stated grounds that hardening it
would flip `shape_snapshots` from loud-crash to silent-skip — a fail-loud-vs-degrade decision
rather than a typo. **That reasoning was wrong, and the error is instructive:** it enumerated two
options, crash and silence, when the seam already shipped a third. Reporting a malformed polygon as
`degenerate_polygon` is neither a crash nor a silent skip — it is the honest answer, and it was
already representable. A scope decision defended by a false dichotomy is still a scope decision;
the give-away was that the justification appealed to a principle the module's own vocabulary
contradicted.

Fixed in 4.77.1: `polygon_to_spadl` returns an empty `(0, 2)` on an odd length (consistent with its
existing `len < 6` behaviour), and `shape_snapshots` emits a row whenever something WAS published,
so the empty array lands below `MIN_VERTICES` and reads as degenerate. Both pinned by tests
verified RED against the pre-fix code.

## References

* Spec: `docs/superpowers/specs/2026-08-07-goal-map-unification-and-visibility-seam-design.md`
* Plan: `docs/superpowers/plans/2026-08-07-goal-map-unification-and-visibility-seam.md`
* ADR-051 (the D3 orientation defect class), ADR-054 (SB360 degradation + the StatsBomb port),
  ADR-009 (raw primitives ship; composites stay consumer-side), ADR-042 (a coverage denominator
  must never masquerade as a signal), ADR-019 (dtype-safe id comparisons).

---

## Amendment — 2026-08-11, silly-kicks 4.80.0: ADR-051 D3 closed

ADR-055 re-keyed `_gk_influence` onto the `GoalMap` and **deliberately deferred the rest**, recording
`{_defensive_line, _packing}` as the remaining unit. This amendment closes that arc. It is an
amendment rather than a new ADR because ADR-055 already contains the decision — under *"Two
sub-decisions taken by the owner during implementation"* it records that `compute_packing_metrics`
*"would have needed a `GoalMap` too — three more breaking changes and ~26 files"*. A new ADR would
restate it.

What the amendment must carry is what ADR-055 does **not**: the fate of Gate B, the scope predicate,
the per-site mechanism split, and the D3 pin's new form.

### The scope was SIX sites, not the two recorded — and the fix is a PREDICATE, not a list

The list ratcheted **2 → 4 → 6** across three plan revisions. Every expansion came from the same
cause: scope was ENUMERATED. It is now bounded by a rule a machine re-runs — a site is in scope iff
it **CALLS** `same_id`/`ids_match` with `home_team_id` — and the pin asserts that rule finds nothing.

Its predecessor is instructive and is recorded so it is not reinvented: a shape-matching predicate
("a `same_id` result guarding a pitch-constant subtraction or a reversing slice") was implemented and
RUN, and it missed **3 of 8** sites including `_defensive_line.py`'s own — because that site decides
direction by sorting from the other end (`argsort(xs)` vs `argsort(-xs)`) without ever reflecting a
coordinate, and its `-xs` is the unary negation the predicate nominated as its EXCLUSION criterion
for score sites. A module-population pin under it would have reported `_defensive_line.py` **already
clean before any re-key**. **Match the SOURCE of the decision, never its downstream shape.**

Corollary, because it is what forced the bad predicate: **a hand-maintained list of FILES TO SCAN is
not a hand-maintained list of EXEMPTIONS.** Narrowing the scan never hides a violation inside it;
only an exemption list can wave one through.

### The mechanism is PER SITE, and ADR-055's `goal_map` ruling does not generalise

| Serves | Takes | Sites |
|---|---|---|
| BOTH teams | `goal_map` | `_defensive_line`, `_packing` |
| ONE team | a bool from `acting_team_attacks_rtl` | `_structural_pass`, `_line_breaking`, `_off_ball_runs`, `_player_influence` |

ADR-055 chose `goal_map` for packing on a packing-specific ground: supplying a float end for the
DEFENDING team requires `0.0 if same_id(...) else 105.0`, the exact fork this arc removes. That
argument has no force at a site needing one team's direction. `acting_team_attacks_rtl` is the
repo's single orientation authority (ADR-028/041) with 7+ production call sites, and ADR-042 already
aligned TF-4 onto it; threading a map into a one-team site would REVERSE that consolidation. At
`_player_influence` it is concrete: the site reflects a GRID, and a map returns a pitch-x the
function would collapse to a boolean on its first line.

**Rule for the next author: functions serving ONE team take the bool; functions serving BOTH take the
map.** The bool is resolved ONCE at the aggregator edge and threaded down — per-frame geometry never
receives the resolver.

### One unresolved-end policy, expressed twice — the helper became nullable too

An earlier revision of this amendment recorded the split as permanent: map sites REFUSE, helper
sites inherit `acting_team_attacks_rtl`'s `False` default. That was decided when the question was
scoped to `_player_influence` alone, and it does not survive contact with the rest of the cycle. A
`GoalMap` that returns `None` and a direction helper that returns `False` are answering the SAME
question with opposite honesty, and the helper's answer is the dishonest one: **a resolved
left-to-right team and a team with no resolvable direction were the same value**, so no consumer
could distinguish them however carefully it was written.

So `acting_team_attacks_rtl` now returns `dtype="boolean"` with `<NA>` for unresolved. ADR-028 D2
had already made the condition audible; this makes it *representable*, which is the difference
between a warning a consumer may ignore and a value it must handle. `.fillna(False)` remains
correct at many of the 21 call sites — it just has to be WRITTEN, with a reason, which is the whole
distinction between a considered default and an inherited one.

**The cost was paid up front and is the argument for the change.** Converting the consumers forced
each to state its policy, and two of them turned out to have none. `_unresolvable_direction_mask`
was a second hand-rolled resolvability test that disagreed with the authority in both directions
(the `astype(bool)` string-qualifier trap, plus a raw-tuple index lookup that misses across
dtypes); it is deleted, because the `<NA>` contract removes the reason a consumer would re-derive
this at all. And the resolver's own `frames["is_ball"].astype(bool)` had been selecting NO player
rows for every provider emitting a string `is_ball` — the ADR-028 defect firing on a whole input
class, invisible precisely because the fall-through returned all-`False`.

**Per-consumer policy, where it is not simply `.fillna(False)`:** `_player_influence` blanks its
three xT columns and KEEPS `reachable_area*`, because the latter is exactly invariant under the
flip (measured: max |delta| 0.0 across 20 players, against 1.17e3 for `off_ball_xt`);
`add_space_creation` refuses the whole row, because its two columns are EXCHANGED rather than
degraded, and half of an exchanged pair is not a partial answer; the shared action-context nulls
the sampled geometry so all eight kernels behind it inherit one decision; and
`resolve_gk_geometry` skips the tracking tier and records the fallback in `*_coord_source`, which
is the honest reading of "this coordinate could not be derived" rather than a clamp applied in an
unknown frame.

**`_player_influence` is the exception and it was MEASURED, not assumed.** The helper's justification
— *"such actions produce NaN geometry anyway because they cannot link"* — was argued for off-ball
runs and does **not** transfer to an xT grid, which exists whether or not any action links. A planted
unresolvable scene showed real numbers emitted on a guessed orientation, so that consumer now blanks
its xT columns. Only the xT columns: `reachable_area*` comes from pitch control, never touches the
reflected grid, and is correct regardless of direction.

The corpus could not answer this — 3 SkillCorner matches, 3,645 actions, **0 unresolvable**. A sample
that cannot produce the failure has not cleared it, so the case was BUILT.

### Gate B is retired for these entries; Gate C replaces its DETECTION — for four of six

Gate B varies `home_team_id`; once the parameter is gone the entry skips on `role="unused"`. Gate C
holds the frames fixed and swaps the MAP.

**But Gate C only applies to the four map consumers.** The two bool sites take no map, so swapping
one moves nothing and such an entry would PASS BY IGNORING ITS INPUT — vacuous by construction, and
the completeness gate's `declared − observed` half rejects it. Their detector is a **behavioural
direction-invariance test**: mirror the FRAMES, hold `home_team_id` constant, require action-LTR
geometry to be unchanged. That test sees the defect AND survives the fix, which neither Gate A (it
swaps the id too, restoring the assumed invariant) nor Gate B (vacuous after the fix) can do.

`gate_c_must_move` is now checked for **completeness**, not just satisfaction: an undeclared witness
fails. Without that, a hand-picked subset lets a partial re-key ship green — and it nearly did.
`packing_goal_threat` is the ONLY witness for packing's back-line site and was almost dropped as a
dead column, because it is constant `0` on the base leg. `0` is the CORRECT answer there; flipping
the end moves it to `[4, 1, 1, 1]`.

**A detector's liveness is not "does it vary across rows" but "does it move when the thing it detects
changes."** A base-leg constancy screen classifies `packing_goal_threat` identically to
`back_n_count`, which genuinely cannot move — and they need opposite verdicts.

### A dead parameter is removed WITH the use, or it is never removed

The re-key exposed a second, larger population: **8 signatures carrying a `home_team_id` that
nothing read** (25 once the cascade below finished; 62 across the whole cycle) — residue from the ADR-028/041 re-keys, which removed the *use* and left the
*parameter*. Cleaning them is not cosmetic. A declared parameter is a claim that the value matters;
eight public functions were making that claim falsely, and the very ADR text above tells the next
author that direction never comes from identity.

They had to be driven to a **FIXPOINT**, because the dead ones formed forwarding CHAINS: the obso
family existed only to hand the argument to `_precompute_obso_lookup`, which ignored it, so
removing the sink killed five public signatures above it, then `_run_values_at_actions`, then
`add_off_ball_run_values` / `off_ball_run_value_xfns` and the atomic mirrors. Three iterations to
converge.

**Two detector limits, both hit, both worth knowing before the next such sweep.** A reads-counter
calls a parameter LIVE when its only use seeds an unused closure default (`_htid=home_team_id`) —
that is how `pausa_xfns` survived the fixpoint. And a call-site sweep keyed on the callee's NAME
misses ALIASED forwarding (`_std = tracking.add_xt_gk`), which left all four atomic mirrors raising
`TypeError` against a target that had already been cleaned. Resolving import aliases and checking
by `inspect.signature().bind()` catches both; neither catches a kwargs DATA TABLE
(`("obso_xfns", dict(home_team_id=...))`), because that is a dict literal, not a call. The suite is
the only backstop that saw all three layers.

**Verify a fence before removing it, and expect per-site answers.** `add_xt_gk` / `xt_gk_xfns`
documented theirs — *"accepted for GK-feature-family signature parity"* — and it was measurably
stale: this very ADR re-keyed two of that family off the parameter, so parity meant matching the
minority, and specifically matching `add_ghost_gk`, which READS its copy. Removed. But
`_off_ball_runs_kernel` KEEPS its unread copy, because its Gate B green **is** the standing
measurement that the parameter is unread, and so does `_compute_space_creation_for_action` (the
recorded "retire by disuse, not removal" case). Those two are why `add_off_ball_runs` and
`add_space_creation` keep theirs as well.

**No value moves**: every removal was AST-verified unread first, so this is signature-only — no
golden shifts, no retrain question, no re-materialization.

### Consequences

* **BREAKING**: `home_team_id` removed from 5 per-frame functions, 6 `add_*` surfaces and their
  `*_xfns`, across `tracking` / `atomic` / `calibration` / `causal` — **and from 11 per-Series
  helpers in `features.py`** the scope predicate could not see, plus the **25 dead-parameter
  signatures** above. 62 signatures in total, across 82 source and test files. Those helpers never *call*
  `same_id`; they declared the parameter and forwarded it, so the predicate that correctly bounds
  the DEFECT does not bound the API MIGRATION. **The two are different sets and the second is
  strictly larger.** Enumerate a removed parameter by signature diff against the base commit.
* **BREAKING**: `acting_team_attacks_rtl` returns `dtype="boolean"`, not `bool` (above).
* **No re-materialization owed for conventionally-oriented frames.** MEASURED, not inferred: every
  re-keyed aggregator was run at the pre-re-key commit and at this one against the same
  home-attacks-right scene — **15 columns, 4 aggregators, all identical**. Where frames are oriented
  otherwise, away-team geometry moves from a wrong value to a correct one.
* **UNORIENTED frames now yield NaN where they used to yield a confident wrong number.** This is
  the `<NA>` contract's only behavioural cost and it is a real one: a consumer feeding absolute
  frames loses columns it used to receive. It is not a regression — those values were mis-projected
  for roughly half the actions — but it will look like one to anyone who has not adopted ADR-029's
  `orient_frames_to_ltr`. The library says so loudly (`OrientationUnresolvedWarning`) rather than
  only in the value.
* The D3 pin is renamed `test_no_module_infers_direction_from_team_identity` and asserts its
  population is EXACTLY empty over eight modules. **Empty is the correct steady state** — stated in
  the docstring so a future reader does not "fix" it by repopulating the set.
