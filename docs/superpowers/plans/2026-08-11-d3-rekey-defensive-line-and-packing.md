# Finish the D3 re-key: every identity-keyed direction site — Implementation Plan

**Status:** DRAFT rev 6, for review. Nothing committed.
**Scheduled:** owner, 2026-08-11.
**Predecessor:** ADR-055 re-keyed `_gk_influence` (4.77.0) and left the rest. **Origin:** ADR-051 D3.

**Revision provenance.** rev 1 → review (3 blockers, 4 errors) → rev 2 → review (3 blockers,
2 majors) → rev 3 → review (3 blockers, 3 majors) → rev 4 → review, execution-order lens (1 execution
blocker, 6 ordering, 2 design answers) → rev 5 → **review, executor-readiness lens: one blocker —
D9 implemented, run, and FAILED** → **rev 6**. Every finding independently re-verified against source
before adoption; D12 was verified BEFORE adoption. § Self-review carries what each revision got wrong.

**All decisions taken (owner, 2026-08-11).** Rev 5 is reorganised around the **commit sequence**,
because the flat task numbering was itself the defect the execution-order review found: three tasks
instructed "do this first" from positions 1, 5b and 7.

---

## Decisions

| # | Decision |
|---|---|
| D1 | Hard break + **behavioural red tests written FIRST** — at the **AGGREGATOR** level (E2) |
| D2 | ~~Extend the canonical scene~~ — **SUPERSEDED by D7** |
| D3 | `_structural_pass.py:146` in scope (2 → 4 sites) |
| D4 | `allow_guess=True` then an explicit `None` refusal, per `_cover_shadows.py:712-716` |
| D5 | Keep the D3 pin; assert its population is EXACTLY empty — **sharpen BEFORE widen** |
| D6 | Gate C for **every** `defect_b=_D3` entry — six |
| D7 | Fix the SHARED `canonical_scene` + add an optional per-entry `scene` seam |
| D8 | `_line_breaking.py:234` + `_player_influence.py:139` in scope (→ 6); **bound scope by PREDICATE, not list** (the predicate is D12) |
| ~~D9~~ | **DEAD — implemented and RUN at round 5; missed 3 of 8 sites including `_defensive_line.py:225`.** Replaced by D12 |
| **D12** | **The pin matches the CALL SOURCE**: a site is in scope iff it CALLS `same_id(..., home_team_id)` / `ids_match(...)`. Verified recall-complete 8/8 before adoption, and blind to the `:98` dead parameter — which RETIRES C1.6's ordering hazard |
| **D10** | **Commit sequence P0 / C0 / C1 / C2** — D7 ships ALONE, before the re-key. Constraint 1 amended accordingly |
| **D11** | **Per-site mechanism: `goal_map` for the 2 both-team sites, `acting_team_attacks_rtl` for the 4 one-team sites** |

---

## Scope predicate — D9 IS DEAD; D12 replaces it

### ⛔ D9 was implemented and RUN. It fails. (round 5)

D9 proposed: *a `same_id(..., home_team_id)` result guarding an assignment that applies a
pitch-constant subtraction or a reversing slice.* Run against the un-re-keyed repo, where a correct
predicate must find all eight sites — **5 matched, 3 MISSED, 0 false positives** (identical when
widened to `ids_match`, so the call half was never the gap):

```
MISS silly_kicks/tracking/_defensive_line.py:225
MISS silly_kicks/tracking/_off_ball_runs.py:397
MISS silly_kicks/tracking/_packing.py:173
```

**The fatal one is `_defensive_line.py:225` — the module this plan is named after.** It decides
direction **without ever reflecting a coordinate**: its branches are `np.argsort(xs)` vs
`np.argsort(-xs)` (then `np.max` vs `np.min`). Its distinguishing operation is `-xs`, a **unary
negation** — precisely the shape D9 nominated as its EXCLUSION criterion for score sites.

So **D9's discriminator was never *direction vs score*; it was *reflects-a-coordinate vs doesn't*** —
a different partition that aligns on five sites and cuts through the sixth. Because the pin asserts a
MODULE population, `_defensive_line.py` would have been reported **already clean, today, before any
re-key**: the permanent invariant wrong from birth, in the direction that reads as success.

Widening D9 to catch the misses means admitting sign flips (colliding with score sites), comparison
inversions (`>` / `<` at `:397`) and bare argument passing (`:173`) — at which point it degenerates
into "the identity result is used at all", the name-mention predicate D5 was escaping. **D9 cannot be
both structural and recall-complete, for the same reason the semantic form could not be structural.**

### D12 — match the SOURCE of the decision, not its downstream shape

**A site is in scope iff it CALLS `same_id(..., home_team_id)` or `ids_match(..., home_team_id)`.**
After the re-key the correct state of the six modules is not "they reflect coordinates in some
guarded way" — it is that they **do not compute the identity boolean at all**. A bare `ast.Call`
match: trivially structural, **recall-complete by construction** (no downstream shape can evade it),
and immune to a future author inventing a seventh way to branch on the boolean.

**VERIFIED BEFORE ADOPTION — the lesson D9 exists to prove, applied to its replacement:**

| Check | Result |
|---|---|
| Recall over the 8 declared in-scope sites | **8/8, 0 missed** — including all three D9 missed |
| Fires on `_off_ball_runs.py:98` (the dead-but-declared parameter)? | **NO** — a call-match cannot see a bare signature parameter |
| Hits outside the family | **18** — the exemption table (below) |

**⚠ This retires C1.6's sharpen-before-widen hazard entirely.** That ordering existed because a
widened *mention* sweep goes red on `:98`. D12 is blind to `:98` by construction, so the evidence
C1.5 protects is safe at any step order.

### Precision comes from SCOPE, not predicate cleverness

**A hand-maintained list of FILES TO SCAN is not a hand-maintained list of EXEMPTIONS.** The first
narrows the gate but never makes an empty result vacuous *within* its scope — a new violation in a
scanned file is still caught. Only the second lets a real violation be waved through, which is what
makes `set()` vacuously true. **D5's binding condition conflated the two, and that conflation is what
forced D9.** Record the distinction in D5.

The gate is therefore three assertions (ADR-056's `_UNDERIVABLE`-asserted-empty idiom):

1. Sweep **all** of `silly_kicks/` for the D12 call shape.
2. Assert **ZERO** hits in the six family modules — the invariant, structural and complete.
3. Assert every hit **outside** them appears in an explicit reasoned table, **with no stale rows** —
   each entry must still match something, so the allowlist is the object under review rather than a
   hidden escape hatch.

**⚠ Budget step 3 honestly: the outside set is EIGHTEEN rows, not the four declared OUT files.**
Measured: `atomic/spadl/utils.py:1132`, `atomic/vaep/features.py:165`, `causal/opportunities.py:335`,
`gkdv/_engine.py:601`, `spadl/orientation.py:219`/`:254`, `spadl/utils.py:1555`,
`tracking/_ghost_gk.py:448`/`:926`, `tracking/_xcross_attempt.py:373`/`:771`,
`tracking/direction.py:169`/`:252`/`:378`, `tracking/kloppy.py:133`, `tracking/utils.py:178`/`:288`,
`vaep/features/core.py:205`. This is real work and was not visible while only four files were checked.

**ALL 18 CLASSIFIED 2026-08-11 — the partition HOLDS, and scope is stable at six sites.** The first
time scope has been verified rather than asserted; it is the check that would have caught both the
2 → 4 and 4 → 6 ratchets.

| Bucket | Sites | Why out |
|---|---|---|
| Score differential | `_xcross_attempt.py:373`/`:771`, `_ghost_gk.py:448`/`:926` | `sd = raw if … else -raw` — unary negation |
| Membership / roster | `gkdv/_engine.py:601`, `tracking/utils.py:288`, `kloppy.py:133` | no direction decision |
| Orientation layer (row selection) | `direction.py:169`/`:252`/`:378`, `tracking/utils.py:178` | identity SELECTS rows; direction comes from the authoritative flag |
| `play_left_to_right` family | `spadl/utils.py:1555`, `atomic/spadl/utils.py:1132`, `vaep/features/core.py:205`, `atomic/vaep/features.py:165`, `spadl/orientation.py:219`/`:254` | coordinate mirrors, but `home_team_id` is the **parameter of the transformation**, not a proxy for a direction the frames already carry |

**⚠ PRECISION, stated because this plan has overclaimed structurality twice.** The last bucket
*is* a coordinate mirror keyed on identity. It is out by a **semantic** judgment, not a structural
one. **So the gate is a STRUCTURAL SWEEP PLUS A REVIEWED ALLOWLIST — not a purely structural gate.**
That is exactly what step 3 is for (the allowlist is the object under review, and a stale row fails),
but D5's binding condition must be worded to say so rather than claiming a structurality the OUT half
does not have. The IN half — zero hits in the six family modules — **is** purely structural and
recall-complete, and that is the half the invariant rests on.

| IN — six sites | Serves | Mechanism (D11) |
|---|---|---|
| `_defensive_line.py:225` | **both** teams | `goal_map` |
| `_packing.py:145`, `:173` | **both** teams | `goal_map` |
| `_structural_pass.py:146` | acting only | `acting_team_attacks_rtl` |
| `_line_breaking.py:234` | acting only | `acting_team_attacks_rtl` |
| `_off_ball_runs.py:375/:378/:397` | acting only | `acting_team_attacks_rtl` |
| `_player_influence.py:139` | attacking only | `acting_team_attacks_rtl` |

**OUT, by the same predicate:** `causal/opportunities.py:335`, `_ghost_gk.py:926` (score
differentials); `gkdv/_engine.py:601` (membership); `direction.py` / `utils.py` / `kloppy.py` /
`spadl/orientation.py` (the orientation layer, where home identity is the legitimate input).

### Why per-site, not uniform (D11)

The plan's own Architecture rule — *one team → bool, both teams → map* — measured against the sites:
only 2 of 6 serve both. **ADR-055's `goal_map` ruling is packing-specific and does not generalise:**
it turns on supplying a float goal end for the *defending* team, which arises only because packing
also calls `select_back_line_players` for the other team.

`acting_team_attacks_rtl` (`_action_orientation.py:158`) is described in-repo as **"the SINGLE
orientation authority (ADR-028 / ADR-041)"** and had **7 production call sites** when ADR-042 aligned
TF-4 onto it. Threading a `GoalMap` into one-team sites would **reverse that consolidation** and
recreate the divergence ADR-042 removed. Concretely, `_player_influence.py:139` reflects a *grid*
(`threat_grid[::-1, ::-1]`) — a `GoalMap` returns a pitch x that the function would collapse to a
boolean on its first line.

**⚠ THE HELPER IS CALLED AT THE EDGE; THE SITE TAKES A BOOL.** `acting_team_attacks_rtl(actions,
frames) -> pd.Series` is per-**ACTION**; the four one-team sites are per-**FRAME** geometry
(`compute_packing_metrics(frame, ...)`, `compute_player_influence(frame, ...)`). Passing `actions`
into per-frame geometry to call the helper in situ would be a **worse** ISP violation than the
`GoalMap` D11 rejects. The aggregator calls the helper ONCE and threads the resulting **boolean**
down — exactly what Architecture already says. Read the table's "Mechanism" column as *"a bool derived
from `acting_team_attacks_rtl` at the aggregator"*, never *"call it here"*.

**⚠ `_line_breaking.py:234` runs the helper's contract BACKWARDS, and it is safe only by an argument
nobody has written down.** The helper's docstring describes flipping frame-sampled positions to land
in the **action-LTR** frame; `_line_breaking:232-234` does the opposite conversion (*"Convert SPADL
action coords to tracking coords"*). The two coincide **only because `(x,y) -> (105-x, 68-y)` is an
INVOLUTION** — its own inverse (`_geometry.py:9` records the rotation). **State this at the site**, or
a future reader will find the contract and the use opposed and "fix" something already correct.

**⚠ `_player_influence.py:139` branches on `attacking_team_id`, not the acting team.** Confirm at
`features.py:4045` and `:4445` that the two coincide for this aggregator before substituting an
acting-team helper; if a frame's attacking team can differ from the action's team, the substitution is
**not value-preserving**.

**⚠ Two unresolved-end policies now coexist, and they COMPOSE on a single data path.**
`acting_team_attacks_rtl` defaults to `False` (no flip) — pinned by
`test_off_ball_runs_orientation.py`, justified as *"consistency, not correctness"* — while Constraint
5 mandates REFUSE for the map path. The split is: map sites REFUSE, helper sites inherit the helper's
default. **But `n_attackers_behind_line` and `line_break` are produced in `_line_break_kernel`, which
consumes `compute_defensive_line`'s output (`:296`, the REFUSE path) AND branches on the acting team's
direction (`:375`/`:397`, the default-False path)** — one output column with two unresolved semantics
at once.

**Rule: where a helper-path decision composes with a map-path value in the same derived column, the
composite row carries the MAP path's NaN.** The refusal wins downstream; the helper's default must
never silently produce a numeric row that a refusal would have blanked. Otherwise a row that should be
NaN is emitted as a plausible number — the original defect class in a new location.

---

## Global Constraints

1. **Gate C MUST land in the same commit as the re-key it detects.** **Amended (D10): fixture
   infrastructure (D7) MAY precede the re-key without violating this** — D7 is not a detector paired
   with a fix; it is the ground the detectors stand on. Every Gate C entry still ships in C1.
2. `gate_c_must_move` must NAME columns, and they must be **MEASURED** to move
   (`test_mirror_registry.py:325`, `:340`). Gate C takes no magnitudes; it asserts `> 1e-12`.
3. Gate C proves the map is CONSULTED, not that the right accessor was chosen.
4. Gate C is BLIND to any path the registered aggregator does not call.
5. **An unresolved goal end REFUSES on the `goal_map` path.** `GoalMap.get -> float | None`
   (`_gk_resolve.py:417`); `None == 0.0` is `False`, so `get(...) == 0.0` silently means "defends
   x=105" — verbatim precedent `_cover_shadows.py:714`. Per-frame functions raise
   `GoalEndUnresolvedError`; the `add_*` edge catches by name and emits NaN. **Helper-path sites
   follow the helper's own documented policy** (see D11).
6. ADR-055's three rules bind for the map sites. `add_*` keep a stable signature
   (`goal_map=None`, self-building) — which is what makes D1's red tests honest (E2).
7. **ADR-059 governs every survivor set here.** Dropping non-moving columns and concluding from the
   remainder is filtering out non-discriminating evidence. Each entry states what makes its set
   discriminating — **or records the shortfall**.

---

## Verified state and measurements

| Fact | Evidence |
|---|---|
| `select_back_line_players` already takes `defends_x0: bool` | `_defensive_line.py:20` |
| `_off_ball_runs.py:375/:378/:397` sit in `_line_break_kernel` (`:268`), NOT `_off_ball_runs_kernel` (`:94`) | function boundaries |
| `defect_b=_D3` at **6** entries | `defensive_line_and_breaks.py:79/:106/:171`, `influence_family.py:160`, `shape_and_structure.py:143/:183` |
| `_line_breaking.py` ward path has **NO detector** — neither Gate B nor Gate C | `add_line_break` entry comment |
| `acting_team_attacks_rtl` = the single orientation authority, 7 prod call sites at ADR-042 | `_off_ball_runs.py:200-211` |

### Gate C carrying capacity — MEASURED on the current scene

| Entry | Carrying | Needs D7? |
|---|---|---|
| `add_defensive_line` | 5 / 6 | no (`back_n_count` dead, documented) |
| `add_packing` | 2 / 4 | no (D7 raises it to 3) |
| `add_structural_pass` | 3 / 3 | no |
| `add_player_influence` | **3 / 7** | **no** |
| `add_line_break` | 1 / 2 | **YES** |
| `add_off_ball_context` | 1 / 6 | **YES** |

**`add_player_influence`'s four non-movers are CORRECT, not degenerate** — `reachable_area*` and
`actor_reachable_area_m2` are pitch-control-only and do not depend on the grid reflection. **Say so in
the entry comment**, or Constraint 7 reads a 4-of-7 non-mover as a filtered survivor set.
`off_ball_xt_diff` moves **6929.71**, corroborating the entry's own recorded *"6.93e3 movement the
identity-keyed grid reflection produces"* — independent evidence the `home_team_id`-flip proxy
measures the intended quantity.

**⚠ `add_line_break` and `add_off_ball_context` both rest on `n_attackers_behind_line` — one column,
two entries, i.e. ONE detector.** Under Constraint 7 that is not discriminating. D7 must revive their
other columns; see the satisfiability check in C0.

### Migration surface — three packages

`tracking/_kernels.py` (2 calls), `causal/_confounders.py` (1), `tracking/features.py` (0);
`calibration/_features.py:224`, `:353`; `atomic/tracking/features.py:154`, `:261` (mirrors) and
`:174`, `:309` (`*_xfns`); `calibration/_vaep_brier_objective.py`. ADR-055 sized its equivalent at
~26 files across two packages — reconcile against that. The migration pattern is already written at
`calibration/_features.py:230`; follow it.

---

# Execution sequence (D10)

## P0 — measurement and decisions. NO COMMIT.

**P0.1 — RUN 2026-08-11 on the REAL corpus. Result: 0 unresolved → C2 probably does NOT exist.**

4 real SkillCorner matches (`1886347`, `1899585`, `1925299`, `1953632`), **`tracking_limit=None`
(FULL frames)**, 16 `(game, period, team)` groups: **`get` unresolved 0 (0.0%), `attacked_goal`
unresolved 0 (0.0%)**. Every group landed in `resolved` — none `guessed`, none `unresolved` — so the
result does not depend on D4's `allow_guess` choice.

**Method note that is load-bearing:** no `tracking_limit`. A capped load starves derived-GK detection
and would have biased this toward "unresolved", manufacturing the very result under test; ADR-055
rule 1 also requires the map be built once per match from FULL frames.

**This independently confirms ADR-055's central claim.** CLAUDE.md records `attacked_goal` as
unresolvable for **35.7%** of SkillCorner team-FRAMES — but that is the PER-FRAME map ADR-055
rejected. The PER-MATCH map resolves **100%** on the same provider. Different estimator, different
answer.

**Reading, stated as a sample and not a clearance:** 4 matches of the sparsest provider available on
pining (metrica is not a pining provider; SkillCorner is the sparse case). **Byte-identical after the
re-key is REACHABLE**, so C2 is probably unnecessary — but C1 must still verify no values moved rather
than assume it, and if any did, C2 splits out per D10.

*(historical: this was Task 7, which said "before writing code" from position 7)*
Measure `n_resolved` vs unresolved `(game, period, team)` on the real corpus. Constraint 5 makes this
decisive: once unresolved ends raise, unresolved rows become NaN, so **"byte-identical" is reachable
only where the map fully resolves.** Minutes of work. **Its outcome decides whether C2 exists**, so it
must precede sequencing, not merely coding.

**P0.2 — RUN 2026-08-11. Result: NOT DECIDABLE IN ADVANCE → the criterion is NOT binding.**

D7 required `add_line_break` to carry more than the shared column. Its only other invariant column is
`line_break`, a boolean the registry records as flipping *"only under the nonsense id … exactly the
`same_id(x, home) else …` branch a two-team swap leaves looking correct."*

**Gate C's perturbation is SYMMETRIC by design and cannot be made otherwise.** `_flip_map`
(`test_mirror_registry.py:285`) swaps BOTH teams' ends, because *"swapping both teams is not the same
as corrupting the map: the result still says the two teams defend opposite ends, so `attacked_goal`
resolves and the degeneracy guard does not fire."* An asymmetric map (one team only) would be
incoherent, fire the guard, and destroy the gate's rival-hypothesis property. **So Gate C's flip is
exactly the two-team-swap shape under which `line_break` is recorded as not moving.**

**Why this could not be closed, and the limit is the interesting part.** The available probe measures
the CURRENT identity-keyed implementation under a `home_team_id` swap; Gate C measures the RE-KEYED
implementation under a symmetric map flip. Round 1 licensed that proxy because the four dead columns
fail through **fixture degeneracy**, which transfers. `line_break` fails through **branch
cancellation**, which does **not** — and the re-keyed code does not exist yet. *A proxy is licensed by
the MECHANISM of the failure it stands in for, not by having been used before.*

**Decision: the criterion is SOFTENED, not dropped** — *"carry more than the single shared column
where achievable, with the shortfall recorded per entry under Constraint 7."* Re-check `line_break`
after C1.2 exists; if it still cannot move under a symmetric flip, record that `add_line_break` and
`add_off_ball_context` share one detector and say so in both entry comments rather than implying two.

**P0.3 — Decide the ADR (E5).** Amend ADR-055; do not mint a new one — it already contains this
decision pre-authored (*"would have needed a `GoalMap` too — three more breaking changes and ~26
files"*). **Decide now, WRITE at C1 commit-prep**: the amendment must record the pin's final form and
D7's outcome, neither of which exists yet.

**P0.4 — RUN 2026-08-11. Result: UNEXERCISED on the corpus → the justification is UNTESTED, and the
case must be BUILT rather than waited for (moves to C1).**

Restated first as a measurement, because as originally written (*"re-verify the 'NaN anyway'
justification"*) it named no falsifiable observation: **count actions where
`acting_team_attacks_rtl` returns its default because direction is unresolvable, and check what
`add_player_influence` EMITS for those rows.**

Measured on 3 real SkillCorner matches: **3,645 actions, 0 unresolvable (0.0%)**. Frames lacking a
`team_attacking_direction` label: **4.3%** — which does NOT propagate to actions, because the lookup
needs only one labelled row per `(game, period, team)`. The helper is not degenerate either:
`flip_true` is 44-48% of actions, so it discriminates rather than defaulting everywhere. **The
ADR-051 PR-4 concern does NOT reproduce** — that cycle recorded the label null on 100% of SkillCorner
rows; through the pining path it is 4.3%.

**Zero unresolvable actions means the failure mode never occurred, so this sample CANNOT falsify the
justification — it is UNTESTED, not confirmed.** The script exits without scoring in that case rather
than reporting a green result: "0 problems found" on a sample that cannot produce the problem is the
vacuous-gate pattern this cycle exists to remove.

**C1 TASK (new, from this result): PLANT the case.** Construct a synthetic scene where a
`(game, period, team)` has no resolvable direction while its actions still link to frames, and assert
what `add_player_influence` emits. A number means the helper's default silently produced a value
where a refusal would have blanked it, and that site takes Constraint 5's refusal instead of the
helper's `False`. The corpus cannot supply this case; building it is the only way to answer.

**P0.5 — Downstream private-consumer check: DONE, CLEAN.** CLAUDE.md mandates checking
`docs/PRIVATE_CONSUMERS.md` before touching any `silly_kicks/**/_*.py`, because path pins fail
**silently** — a renamed module degrades a consumer's guard with no `ImportError`. Checked
2026-08-11: **no pins on any of the six modules, nor on `_action_orientation`.** Recorded so the next
reader does not re-derive it. Note this covers PATH pins; the six modules are private, so a consumer
importing `compute_defensive_line` directly is outside this file's scope and outside the repo's
ability to see.

## C0 — D7 alone. TEST INFRASTRUCTURE ONLY; zero production change.

1. Make `canonical_scene` non-degenerate: real inter-frame motion, and a pass sequence producing a
   secured reception.
2. Add an optional per-entry `scene` on `MirrorEntry`, defaulting to `canonical_scene`.
3. **Re-derive every drifted literal.**

**Acceptance:** `packing_secured` non-NA on ≥2 rows with >1 distinct value; detectable off-ball
displacement. **The `add_line_break` / `add_off_ball_context` criterion is SOFTENED per P0.2** — carry
more than the single shared column *where achievable*, and record the shortfall per entry under
Constraint 7 where not. It is **not** binding, because it was measured to be undecidable before the
re-key exists, and a binding-but-undecidable criterion would stall C0 with four entries registered.

**Why C0 is its own commit — this is enforcement, not style.** D7's rule is *"every drifted literal is
a FINDING requiring re-derivation, never a rebase."* **That distinction is only reviewable in a diff
where nothing else moved.** Inside a three-package refactor, a rebase wearing a finding's clothes is
invisible: the reviewer cannot tell a literal that moved because the scene changed from one that moved
because the re-key changed a value.

**Known movers — NON-EXHAUSTIVE** (two of six entry files): `add_team_shape` 20.2 m / 44.0 m;
`add_shape_graph` edge count 16 (Delaunay connectivity is a function of the point set);
`add_packing`'s "exactly 0.0 on the two in-domain pass rows"; the five `add_defensive_line` deltas
(23.75 / 11.0 / 3.0 / 4.0 / 6.0). **Also re-derive `add_player_influence`'s `tol=1e-6`** — justified by
an ABSOLUTE residual against ~8e3-magnitude columns, so more players and motion move the
justification even if the tolerance still passes. Guard: full mirror suite +
`test_pr5_chirality_gates.py`.

## C1 — the re-key. One commit.

**C1.1 — Red tests FIRST, at the AGGREGATOR level (D1, E2 — with E2's stated reason CORRECTED).**

**⚠ "The same tests go green unchanged" is FALSE at BOTH levels, and rev 5 initially repeated the
error one level up.** Measured: `home_team_id: int | str` is a **required** keyword argument on
`add_defensive_line` (`features.py:1200`), `add_packing` (`:1349`) and `add_player_influence`
(`:4361`) — no default. Constraint 6 removes it, so
`add_defensive_line(actions, frames, home_team_id=H)` must become
`add_defensive_line(actions, frames)`. The invocation changes at the aggregator level too. **Any
hard break edits the call; only a shim avoids that, and D1 rejected the shim on stronger grounds.**

**The honest claim is that the ASSERTION is unchanged, not the invocation.** That is what TDD's
red→green actually rests on — the expectation must not move. So: keep the assertion byte-identical
across the transition, adapt the invocation mechanically, and **carry the red output AND the
red→green diff in the PR body at every level**, so a reader can verify the assertion was not
rewritten to fit the new behaviour.

**Write them at the aggregator level anyway**, for E2's two surviving reasons: it is the only level
where Constraint 5's NaN-row policy is *observable*, and where D4's `allow_guess`-then-refuse choice
shows up in output rather than in a raise.

Assert the failure **MODE** — "N FAILED" is satisfiable by an import error or a `TypeError`.

**C1.2 — Re-key all six sites (D11 mechanism per site).** `_packing` and `_structural_pass` are
duplicated *by design* — same spelling; a shared helper is a welcome outcome, not a speculative goal.
`_off_ball_runs:378` is a **correctness coupling**: it un-mirrors `defensive_line_x` assuming the
producer emitted home-attacks-right, so re-keying the producer while leaving it *moves* the bug.

**C1.3 — `add_*` edge catches (ATOMIC with C1.2, E4).** `except GoalEndUnresolvedError` exists at
`features.py:3166/:3642/:3826/:3967` — all *other* aggregators; **none of the in-scope ones has it.**
**C1.2 and C1.3 cannot be separately committed or separately verified:** between them, every
aggregator with an unresolved end propagates the exception to its caller, including the calibration
pipeline and the atomic wrappers. Anyone running the suite in between sees failures caused by the
plan, not the code.

**C1.4 — Migrate the caller surface** (three packages, above).

**C1.5 — Gate C for all six entries.** Four are D7-independent and could have been registered at C0
(`add_defensive_line`, `add_packing`, `add_structural_pass`, `add_player_influence`); all six land
here so each ships with the re-key it detects (Constraint 1). Per Constraint 7 each entry comment
states what makes its set discriminating — including `add_player_influence`'s correct non-movers.

**`add_off_ball_runs` is NOT in this list and its Gate B entry is preserved as-is.** `home_team_id`
reaches `_off_ball_runs_kernel:98` and is never read (ADR-042), so its GREEN *is* the measurement that
the parameter is dead; declaring `"unused"` makes Gate B skip and throws that away. **If this cycle
strips the parameter, relocate the evidence — do not drop it.** Interacts with C1.6 step 2.

**C1.6 — The D3 pin, on D12. The ORDERING HAZARD IS RETIRED.**

Rev 5 required sharpen-before-widen because a widened *mention* sweep goes red on
`_off_ball_runs.py:98`'s dead-but-declared parameter, and the obvious fix would delete the evidence
C1.5 protects. **D12 is a CALL match and cannot see a bare signature parameter — verified: its
`_off_ball_runs.py` hits are `:375` and `:397` only.** So step order no longer carries that hazard.

1. Rewrite the predicate to D12 (`same_id`/`ids_match` **called** with `home_team_id`).
2. Widen the scanned set from three hardcoded files to **all of `silly_kicks/`**, asserting ZERO hits
   in the six family modules.
3. Add the reasoned exemption table for the **18** outside hits, asserting no stale rows.
4. **Delete the contradictory `assert reads`** (`:434`, asserts NON-empty). Keep
   `assert rekeyed not in reads` (`:429`) — it generalises once the set widens.

**Binding conditions, with D5's conflation corrected:** the derivation stays STRUCTURAL (D12 makes
this trivially true); the docstring states EMPTY is the CORRECT steady state; rename only once the
predicate is D12 — the name must not outlive the predicate. And record the distinction that forced
D9 in the first place: **a hand-maintained list of FILES TO SCAN is not a hand-maintained list of
EXEMPTIONS** — only the latter can wave a real violation through and make `set()` vacuous.

**C1.7 — Accessor correctness, registry, stale anchors.** Extend `test_goal_map_consumers.py`
(`attacked_goal` must be a REAL lookup, never `105.0 - get(...)`). Revisit both
`conftest_id_scalar.py` registrations. **Fix three stale anchors**:
`defensive_line_and_breaks.py:11` cites `_defensive_line.py:73` as identity-keyed when `:73` is now a
docstring example of the CORRECT form; `test_mirror_registry.py:399` cites `:210` for the inference at
`:225`.

**C1.8 — Docs, version.** `4.80.0` in all five sites. Write the ADR-055 amendment (decided P0.3).
C4 count unchanged — verify.

**DELETE the `Finish the D3 re-key` row from `TODO.md` outright** — do not rewrite it. This cycle IS
that row's work, so on landing it describes something done; the outcome belongs in the CHANGELOG, and
a tracker row for completed work is just a second place for the truth to rot.

> **This step was LOST and had to be restored.** Rev 4 said *"discharge the TODO row"*; by rev 6 that
> had become *"rewrite the pending TODO.md edit to final scope"*, and `grep -i discharge` over the
> plan returned nothing — an instruction to RETIRE the row had silently become an instruction to
> PERPETUATE it. Acting on it, the row was rewritten three times mid-cycle, each time re-deriving
> guidance for an executor who will never exist, and twice going stale against this plan in the
> process (it carried the dead D9 predicate, then the `goal_map.get(...) == 0.0` fail-open). The
> `TODO.md` edit was reverted entirely: **if the work is done the row is removed, and if it is not
> done there is no point rewriting a row that is about to be removed.** Same defect class as the row
> itself — a document edited by accretion, carrying two epochs, with the earlier instruction dropped.

**C1 stays ONE commit, and the reasoning mirrors C0's.** C0 must be alone so a moved literal is
attributable to the FIXTURE; **C1 must be whole so a moved literal is attributable to the RE-KEY.**
Splitting C1 reintroduces exactly the ambiguity C0 exists to remove.

**If it ever must split, the only legal seam is PER-SITE-PLUS-ITS-ENTRY** — never "re-key, then
gates", because Constraint 1 binds each Gate C entry to the re-key it detects. That yields six small
commits, with two further constraints: **C1.6's pin cannot land before the last site is re-keyed**
(it asserts an empty population), and **C1.8's version bump must be last.**

## C2 — conditional on P0.1. Re-materialization with provenance.

Only if unresolved > 0. **Must not ride inside C1**, or the CHANGELOG cannot say which of the two
moved the numbers. Downstream (second hop; no symbol sweep returns these):
`docs/research/covariate_invariance/` (via `_confounders.py`) and
`docs/research/tf19_signoff_power/invalidation.json`.

---

## Architecture (for the ADR amendment)

**Direction is resolved once at the edge; the pure core takes resolved direction.**
`select_back_line_players(defends_x0: bool)` IS that core and is already correct.

**Rule: functions serving ONE team take the boolean (`acting_team_attacks_rtl`); functions serving
BOTH take the map.** D11 applies it. Passing a `GoalMap` into a per-frame function remains an
Interface Segregation cost — justified for the two both-team sites, recorded as a trade rather than a
default.

---

## Self-review — what each revision got wrong

**rev 1** — called `_off_ball_runs` a caller when it is a re-key site, citing for its deltas the very
file that says there are three sites; wrote `goal_map.get(...) == 0.0`, a fail-open, 60 lines from a
repo comment saying so; named four columns that cannot move (one not emitted at all); published
occurrence counts as call sites; missed a gate going vacuous while writing a plan about gates going
vacuous; asserted detection-before-fix was impossible with six red detectors in the repo.

**rev 2** — expanded the re-key to four modules while leaving Gate C at two; recommended a pin form
that was **not implementable**, having recommended it without reading the test body; over-corrected
rev 1's inflated caller counts past the true figure, dropping two packages.

**rev 3** — **authored the bounding predicate and never ran it**, leaving two live sites unnamed and a
completeness claim contradicted by the plan's own table; deferred a measurement that changed the plan
when taken; ordered the pin's steps so step 1 destroys the evidence another task protects.

**rev 4** — specified a predicate that must be simultaneously **structural and semantic** without
saying how; left the **commit shape unspecified** while three constraints over-determined it; claimed
red tests would go *"green unchanged"* across a hard break that edits the invocation; generalised
ADR-055's packing-specific parameter ruling to six sites, four of which want a boolean — **while
citing ADR-042 as the reason to preserve one gate and never noticing it as the re-key mechanism**;
numbered three "do this first" tasks at positions 1, 5b and 7.

**rev 5** — **specified D9 and did not run it**, the identical failure to rev 3's unrun predicate, one
revision after naming it as the through-line. Implemented at round 5, it missed 3 of 8 sites
including `_defensive_line.py:225` — **the module this plan is named after** — and would have reported
that module already clean before any re-key. The cause is instructive: D9's exclusion criterion
("a sign flip is unary negation") is the *same shape* as `_defensive_line`'s own direction mechanism
(`argsort(-xs)`), so the predicate excluded an in-scope site by the rule written to exclude
out-of-scope ones. Rev 5 also stated D11's mechanism per SITE when the helper must be called at the
EDGE, and stated the two unresolved-end policies one level too high to see that they COMPOSE inside
`_line_break_kernel`.

**And in requesting round 4 I mis-cited my own document**, claiming Task 9 said "single commit" — rev
1's phrasing, dead since rev 2.

**The through-line, sharpened:** every revision's worst finding came from a claim asserted rather than
executed — a file read for one purpose and not another, a gate recommended without reading its body, a
predicate authored and not run, then a *second* predicate authored and not run. **D12 is the first
scope rule verified BEFORE adoption** (8/8 recall, blind to `:98`), and the verification cost about
two minutes.

**The through-line: enumerating where a defect lives instead of deriving it, and trusting the
document over the source — including when the document was mine.** D9's shape-match is the first form
of the predicate that a machine can re-run, which is the only form that cannot go stale.
