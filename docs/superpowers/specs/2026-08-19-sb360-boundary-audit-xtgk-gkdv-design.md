# SB360 boundary-audit closeout: xtgk.compute_xt_gk_v2 + the three gkdv entry points

**Status:** Draft for review (uncommitted). Supersedes the "structurally blocked" framing in
`TODO.md` and `tests/sb360/test_registry_surface.py::UNAUDITABLE_BOUNDARY`.

## Executive summary (for a reviewer with no prior context)

The SB360 coverage audit (ADR-053) records, for every public aggregator, whether it *fabricates a
plausible-but-wrong number* when handed a velocity-less StatsBomb freeze-frame. Four public
"boundary" entry points outside `tracking.__all__` were parked as **un-auditable** behind a strict
xfail, each with a recorded reason. This spec resolves all four and **empties
`UNAUDITABLE_BOUNDARY`**, retiring the "un-auditable boundary" category entirely.

The `xtgk.compute_xt_gk_v2` reason — *"needs an xG-calibrated MarkovPossessionValue and silly-kicks
ships no xG model, so any port would audit the stub not the library"* — is a **category error**. The
audit measures **velocity/frame degradation** (a cross-leg delta), never **value quality**. Two
patterns already in the registry solve the two sub-objections:

- **injected model** — `audit_xt()` (`scripts/_sb_battery.py:33`) synthesizes a non-degenerate
  deterministic `ExpectedThreat` and injects it into six aggregators; no bundled artifact is
  required.
- **frame-blind aggregator** — `visible_area_coverage` (`scripts/_sb_battery.py:200`) reads no
  frame, synthesizes its non-frame inputs, and *records the honest "the axis cannot reach it"*
  rather than manufacturing a difference. This resolves to the same `identical` -> `works` verdict
  `spadl.add_restart_coordinates` carries — there is no distinct "cannot-reach" adjudication; it is
  prose for `identical`.

`compute_xt_gk_v2` is the intersection: it is **frame-blind** (reads `actions` + marts-derived
`retention_features` + three injected ports; no `frames`, no `vx/vy/speed`), so a deterministic port
double injected identically into both legs cannot "audit the stub" — the stub's values cancel in the
delta. Its honest verdict is `identical` -> `works` (frame-invariant), exactly like
`spadl.add_restart_coordinates` already carries.

The three gkdv entry points are a **different, genuine** situation — gkdv *does* consume velocity —
but they resolve just as cleanly: gkdv sits downstream of `serve_ghost_gk_positions`, which **refuses
velocity-less freeze-frames (ADR-054)**. So on the freeze-frame leg gkdv serves no ghost, scores zero
frames, and produces NaN — the identical refusal `add_ghost_gk` is *already* registered for as
`AxisVerdict("all_nan", "honest_nan")` (`tests/sb360/_entries/_gk.py:22`). gkdv **inherits** that
verdict; it is coherent with the existing ADR-054 story rather than a new claim.

**What emptying the set does and does NOT mean.** These four entries gain *honest, admissible*
verdicts — but two of them are **structurally shielded** from the degradation the audit measures, and
the framing must say so plainly. `compute_xt_gk_v2` is **frame-blind**: its `identical` -> `works` is a
**frame-coupling tripwire** (a regression invariant that flips if a future edit makes it read a
frame), **not** evidence that xt_gk_v2 is velocity-robust or SB360-computable — its velocity-dependence
lives in its *inputs*, computed upstream from tracking (C2 below). gkdv's `honest_nan` is **inherited**
from the upstream ADR-054 refusal; the arms are never reached on the freeze-frame leg, so their
*intrinsic* zero-velocity behaviour is untested and contingent on that refusal (C3 below). Retiring the
*category* (`UNAUDITABLE_BOUNDARY` -> empty) is legitimate cleanup; it must **not** be read as "gkdv
and xt_gk_v2 are now velocity-audited end-to-end." Every recorded verdict below is one of two kinds:
**substantive** (a velocity-*consuming* function whose own handling moved the value) or
**structural/inherited** (frame-blind -> `identical`; downstream-of-refusal -> `honest_nan`). Only the
first is degradation coverage. Both of this cycle's new subjects are the second kind.

**No xG model, no bundled GhostGkModel, and no perturbation of the shared fixture's positions are
required.** Net changes: 4 registry entries + their adapters, `UNAUDITABLE_BOUNDARY` emptied, one
strict xfail retired into a plain passing completeness assertion, `NOT_EXERCISED_BUDGET` raised with
a recorded reason, a `verdict_provenance` field + meta-gate **locking the frame-blind half** of the
substantive/structural distinction (`works`⇒structural) and author-asserting the inherited-refusal
half (Part 4), an ADR-053 amendment, and the stale TODO entry deleted.

## Problem statement

`tests/sb360/test_registry_surface.py` carries two facts:

```python
BOUNDARY_ENTRY_POINTS = {          # tests/sb360/_registry.py:69
    "gkdv.build_ghost_frames",
    "gkdv.delta_das",
    "gkdv.delta_threat_suppression",
    "xtgk.compute_xt_gk_v2",
    "spadl.add_restart_coordinates",   # already registered
}

UNAUDITABLE_BOUNDARY = {           # the four not yet registered, each with a reason
    "gkdv.build_ghost_frames": "...frame-pair shape, needs a fitted GhostGkModel",
    "gkdv.delta_das": "...frame-pair shape, consumes build_ghost_frames output",
    "gkdv.delta_threat_suppression": "...frame-pair shape, plus a fitted ExpectedThreat",
    "xtgk.compute_xt_gk_v2": "...MarkovPossessionValue needs an xG fit; any port audits the stub",
}
```

`test_every_boundary_entry_point_is_registered` is a **strict xfail**: registering any of the four
forces the marker to be revisited (a strict xfail that starts passing fails CI). `test_uncovered_
boundary_points_each_carry_a_reason` additionally asserts that a name is *either* registered *or*
carries a reason, and that **a reason which outlives its need fails** (`stale = UNAUDITABLE_BOUNDARY &
SB360_ENTRIES` must be empty). So registering an entry *forces* the deletion of its excuse.

## The reframe (why the block is not real)

The audit is a two-leg fabrication detector. `run_axis` (`tests/sb360/_harness.py:24`) builds:

- **Leg A** — one freeze-frame per action via the real `snapshot_to_tracking_frames` (velocity
  declared unavailable; `speed_source="unavailable"`).
- **Leg B** — a 10 Hz velocity-bearing neighbourhood with **identical positions at the linked
  anchor frame**.

and compares `out_a[col]` vs `out_b[col]` per emitted column. The verdict is a **cross-leg delta**.
It never certifies whether a value is *good* — only whether the aggregator's number *changes (or
degrades) when velocity is absent*. Therefore:

- A **frame-blind** function's honest verdict is `identical` -> `works` (the value cannot change
  because the function never reads the thing that differs between legs). The injected ports are held
  identical across both legs, so their values are irrelevant to the delta — they are scaffolding, not
  the subject.
- A **velocity-requiring** function that **refuses** velocity-less input produces NaN on Leg A and a
  real value on Leg B -> `all_nan`/`partial_nan` -> `honest_nan`.

## Design

### Part 1 — `xtgk.compute_xt_gk_v2`

**Signature** (`silly_kicks/xtgk/_metric.py:88`):

```python
compute_xt_gk_v2(actions, *, possession_value, retention, turnover_cost,
                 kappa=1.0, pressure_column="pressure", domain_column="is_gk_distribution",
                 pressure_levels=None, retention_features=None, l=N, w=M) -> pd.DataFrame
```

It emits five columns: `xt_gk_v2_position`, `xt_gk_v2_pev`, `xt_gk_v2_retention_loss`,
`xt_gk_v2_dzv`, `xt_gk_v2`. It scores **every finite-coordinate action** (`finite =
finite_coord_mask(actions)`; `domain_column` only drives a warning), reading `actions[start/end_x/y]`
+ `actions[pressure_column]` + `retention_features` + the three ports. **No frame input.**

**Adapter** (`_boundary.py`, inline like `_call_restart_coordinates`):

1. Copy `actions`; add a deterministic `pressure` column (integer terciles derived from
   `action_id`, non-constant so `PressureLevels.apply` exercises >1 level). Leave `is_gk_distribution`
   absent so `_warn_if_unattested` early-returns (its guard is `if domain_column not in
   actions.columns: return`), i.e. no warning to escalate.
2. Build three **live, velocity-blind, deterministic** port doubles (the `audit_xt()` pattern):
   - a `PossessionValue` double whose `value(zone, p)`, `surface(p)`, `delta_v(s, s_next)` return
     non-degenerate, monotone-in-zone numbers (so `position`/`pev`/`retention_loss`/`dzv` are each
     non-constant across the scored rows — the liveness requirement);
   - a `RetentionModel` double whose `predict_proba(features)` returns a deterministic vector in
     `(0, 1)`, non-constant across rows;
   - a `TurnoverCost` double whose `value(zone, p)` returns a deterministic non-degenerate number.
   Plus a real `PressureLevels` constructed on the synthetic `pressure` column and passed
   **explicitly** via `pressure_levels=` — a definite branch, not the `possession_value` double's
   `.pressure_levels` fallback (`compute_xt_gk_v2` hard-raises without it, `_metric.py:109`). And a
   `retention_features` frame that **omits** `COORD_DERIVED_NAMES` so `_check_coordinate_coherence`
   early-returns (`_metric.py:42`, "not a retention-feature frame (e.g. a caller stub)") — the double
   reads only its own synthetic feature column.
3. Call `compute_xt_gk_v2(augmented_actions, possession_value=..., retention=..., turnover_cost=...,
   pressure_levels=..., retention_features=...)`; return the five-column frame.

**What the verdict is, and what it certifies — a frame-coupling tripwire, not degradation coverage
(C1).** Both legs share identical `actions` (both call `_actions_frame()`), the adapter derives its
augmentation deterministically from `actions` alone, and the function ignores `frames` — so `out_a`
and `out_b` are byte-identical finite numbers **by construction**: observation `identical` ->
adjudication `works` (frame-invariant). This is a **tautology, not a degradation test** — a frame-blind
function *cannot* differ across two legs that differ only in the frame. What it earns is a **regression
invariant**: if a future edit makes `compute_xt_gk_v2` read a frame, the audit flips off `identical`
and CI catches it. The live, non-constant columns prove the injected **doubles** are live (so the
`identical` is a real number comparison, not `NaN==NaN`) — they do **not** prove anything about
`compute_xt_gk_v2`'s velocity handling, because there is none to exercise.

**What `works` does NOT mean here (C2).** `compute_xt_gk_v2` is the scorer a reader consults to ask
"is xt_gk_v2 safe on velocity-less SB360?" — and `works` must not be read as "yes." The metric's
velocity-dependence did not vanish; it **moved into the inputs**: `pressure`, `is_gk_distribution`,
and `retention_features` (geometry + pressure), all computed upstream from velocity-bearing tracking.
On real SB360 those inputs are unavailable, so the metric is arguably **un-computable** there. The
audit stamps `works` because the *function* fabricates nothing through a frame it never reads — not
because the *metric* is velocity-robust or SB360-computable. The entry carries a rationale stating
exactly this — **mandatory** for a `structural` boundary entry under the Part 4 provenance gate (the
base vocabulary does not force one on `works`+default tolerance, so the provenance gate is what
requires it).

**Predicted verdict (to be transcribed from execution, per the registry contract):**

| axis / roster | observation | adjudication |
|---|---|---|
| velocity / full | identical | works |
| visibility / gk_absent | identical | works |
| visibility / defender_absent | identical | works |
| visibility / gk_one_end | identical | works |

(Frame-blind, so roster ablation cannot reach it either — the honest reading, same as
`visible_area_coverage`.)

### Part 2 — the three gkdv entry points

**Shapes** (`silly_kicks/gkdv/`):

```python
build_ghost_frames(frames, *, model=None, home_team_id, carrier=None, params) -> (cf_frames, provenance, report)
delta_das(actual_frame, ghost_frame, *, attacking_team_id, params) -> float          # per-frame scalar
delta_threat_suppression(actual_frame, ghost_frame, *, attacking_team_id, xt, goal_map, params) -> float
```

gkdv's native granularity is **per `(game_id, period_id, frame_id)` scored frame**, and the arms
return **scalars**. The per-action harness bridges this via the harness's own `links`: the value **at
the action's anchor frame** — a real library output at a real frame, not an invented aggregate.

**Shared adapter helper** (`_boundary.py`), one thin wrapper per entry point:

1. Build a deterministic `carrier` DataFrame (`[game_id, period_id, frame_id, ball_carrier_team_id]`)
   assigning possession to each action's own team at every frame. This is supplied to
   `build_ghost_frames(carrier=...)` (a first-class parameter, `_engine.py:475`), so possession
   resolves **without** touching the shared fixture's positions — zero blast radius on the other
   locked entries. It is input data the library consumes (like `xt`/`visible_area`), not a fake of
   library behaviour.
2. `cf, prov, report = build_ghost_frames(frames, home_team_id=HOME_TEAM_ID, carrier=carrier)`
   (`model=None` -> default league-average ghost; no fitted GhostGkModel needed — the gkdv engine
   tests run this way).
3. `scored = prov[prov["drop_reason"].isna()]`. For each action, find its anchor frame via `links`;
   if that frame is in `scored`, slice `actual = frames[frame==f]`, `ghost = cf[frame==f]`, restricted
   to the defending keeper's presence, and compute the entry's value; otherwise NaN.
4. Return a per-action frame of the entry's columns.

Per entry:

- **`gkdv.build_ghost_frames`** — emits `ghost_x`, `ghost_y`, `displacement_m` (the defending
  keeper's substituted position + displacement) at the action's anchor frame.
- **`gkdv.delta_das`** — emits `delta_das` (needs the `[das]` extra; the audit already gates
  DAS-dependent entries on it).
- **`gkdv.delta_threat_suppression`** — emits `delta_threat_suppression`, using the synthetic
  `audit_xt()` for its `xt` port and a `goal_map` the adapter resolves **itself** via the public
  `tracking.resolve_defended_goals(frames)`. The engine does not return its map, so the adapter
  reproduces it rather than reaching for a non-existent accessor — and this is byte-identical to the
  map the ghosts were served under, because `build_ghost_frames` pins its map via
  `_pin_defended_goal(frames)`, which *is* `resolve_defended_goals(frames)` (`_engine.py:199`), a pure
  function of the same `frames`. So the arm scores under the same orientation the ghosts were served.

**Why the verdict is `honest_nan` and LIVE:** on **Leg A** (velocity-less), `serve_ghost_gk_positions`
refuses at the shared serving seam (ADR-054), so `build_ghost_frames` serves no ghost, marks every
eligible frame `_DROP_NO_GHOST`, and scores zero — every action is NaN. On **Leg B**, the fixture's
cross (a1), shot (a2) and goalkick (a3) sit within the 35 m domain with the defending GK present, so
those actions score real values. Observation: Leg A entirely NaN, Leg B finite on the in-domain
actions -> `all_nan` -> `honest_nan`, the **same class `add_ghost_gk` already carries**. Live because
Leg B is finite (a real asymmetry, not both-NaN).

**Fixture domain, confirmed against `tests/sb360/_fixture.py`** (home GK x=5 defends x=0; away GK
x=100 defends x=105; ball at action-start at the anchor frame):

| action | type / team | ball@anchor | attacked goal | dist | defending GK | in-domain (full)? |
|---|---|---|---|---|---|---|
| a0 | pass / home | (52.5, 34) | x=105 | 52.5 | away (x=100) | no (>35) |
| a1 | cross / home | (88, 8) | x=105 | ~31 | away (x=100) | **yes** |
| a2 | shot / home | (95, 34) | x=105 | 10 | away (x=100) | **yes** |
| a3 | goalkick / away | (5.5, 34) | x=0 | 5.5 | home (x=5) | **yes** |
| a4 | dribble / home | (60, 50) | x=105 | ~46 | away | no |
| a5 | throw_in / away | (40, 68) | x=0 | 40 | home | no |

**Predicted verdicts (to be transcribed from execution):**

| axis / roster | observation | adjudication | reason |
|---|---|---|---|
| velocity / full | all_nan | honest_nan | Leg A serves no ghost (ADR-054); Leg B scores a1/a2/a3 |
| visibility / defender_absent | all_nan | honest_nan | keepers present; same as full |
| visibility / gk_one_end | all_nan | honest_nan | home keeper kept -> a3 scores on Leg B |
| visibility / gk_absent | no_signal | not_exercised | no keeper at all -> both legs score zero |

`gk_absent` (`no_signal` -> `not_exercised`) is honest and expected: gkdv requires a defending GK, and
`gk_absent` removes both. It raises `NOT_EXERCISED_BUDGET` by the number of gkdv columns on that one
roster (5: `ghost_x`, `ghost_y`, `displacement_m`, `delta_das`, `delta_threat_suppression`), recorded
with that reason.

**What the gkdv verdict certifies, and its contingency (C3).** The `honest_nan` is produced *entirely
upstream*: on Leg A `serve_ghost_gk_positions` refuses (ADR-054), zero frames are scored, and
`delta_das`/`delta_threat_suppression` are **never reached**. So this is a **structural/inherited**
verdict, not a test of the arms' intrinsic zero-velocity behaviour. If an arm internally fabricated on
velocity-less frames (a velocity-blind DAS fallback, say), this audit would **not** catch it as long as
the seam refuses first. The coverage is **contingent on that refusal**: were it ever lifted (as ADR-063
lifted the four pitch-control aggregators to the zero-velocity positional model), the arms' intrinsic
behaviour would need its own probe. Each gkdv entry therefore carries a rationale recording that its
verdict is inherited from the ADR-054 refusal and naming this contingency — **mandatory** under the
Part 4 provenance gate, so the distinction lives test-locked in the registry, not only in this spec.

### Part 3 — retiring the category

Registering all four **empties `UNAUDITABLE_BOUNDARY`** and makes
`test_every_boundary_entry_point_is_registered` pass:

- Delete all four entries from `UNAUDITABLE_BOUNDARY` (forced: `test_uncovered_boundary_points_
  each_carry_a_reason`'s stale check fails otherwise).
- Remove the `@pytest.mark.xfail(strict=True)` from `test_every_boundary_entry_point_is_registered`;
  it becomes a plain passing completeness assertion (a boundary point added later still must register
  or CI fails). Update its docstring and reason string (no longer "four ... structurally out of
  reach").
- `test_uncovered_boundary_points_each_carry_a_reason` stays — it now guards an empty set and still
  fires if a *future* boundary point is added without a reason.
- Raise `NOT_EXERCISED_BUDGET` (49 -> 54, subject to the transcribed count) with the gk_absent reason.
  **`gk_absent` is the sole contributor**: the other two visibility rosters (`defender_absent`,
  `gk_one_end`) score a live `honest_nan` (a keeper is present, so Leg B scores), so no gkdv column is
  `not_exercised` on *every* visibility roster — none enters `columns_exercised_on_no_roster` /
  `_EXPECTED_DARK_COLUMNS` (`_registry.py:262`), and that set is left unchanged. Confirmed by
  execution, not assumed.

### Part 4 — Enforcing the distinction (`verdict_provenance`)

The substantive-vs-structural distinction (executive summary + ADR amendment) is what stops an empty
`UNAUDITABLE_BOUNDARY` being misread as end-to-end coverage. Documenting it in prose + rationales
leaves it **unenforced** — a future edit could strip a rationale, or add a new frame-blind boundary
entry as a bare `identical` -> `works` with no marker, and nothing would fire. This codebase closes
documented-but-unenforced gaps with a registry gate (ADR-056: *a floor cannot detect an omission*), so
the distinction is **enforced where it is machine-checkable** and author-asserted where it is not —
with the ceiling named explicitly below (the Known limit), because overstating "test-locked" is itself
the over-claim this Part exists to prevent.

**Mechanism (minimal, scoped to boundary entries):**

- `VERDICT_PROVENANCE = frozenset({"substantive", "structural"})` in `tests/sb360/_vocabulary.py`. A
  **substantive** verdict comes from a velocity-consuming function whose own handling moved the value;
  a **structural** verdict comes from a function the axes cannot substantively reach (frame-blind ->
  `identical`; downstream-of-a-refusing-seam -> `honest_nan`).
- A per-entry `verdict_provenance: str | None` field on `Sb360Entry` (paired with a
  `provenance_rationale: str | None`) — `None` for the `add_*` surface (where the distinction is not
  at issue), **required** on `BOUNDARY_ENTRY_POINTS`. Per-entry rather than per-verdict because
  boundary entries are uniform; the gate iterates all of an entry's verdicts and asserts none
  contradicts the entry-level token (a future genuinely-mixed entry is a later refinement, noted).
- A meta-gate `test_boundary_entries_declare_admissible_provenance` in `test_registry_surface.py`:
  1. every **registered** boundary entry carries `verdict_provenance in VERDICT_PROVENANCE` (population
     derived from `BOUNDARY_ENTRY_POINTS & SB360_ENTRIES`, asserted exactly — a new boundary entry
     without it fails);
  2. **admissibility from the observation** — a `works` (from `identical`) verdict forces `structural`
     (a value that cannot move was not substantively handled); a `differs_by_design`/`silent_degrade`
     verdict forces `substantive` (the value moved because of the function). `honest_nan`/
     `not_exercised` are human-declared (inherited-NaN and own-refusal-NaN are indistinguishable from
     the observation), but the field must be present;
  3. a `structural` entry MUST carry a non-empty `provenance_rationale` (a dedicated per-entry field,
     so the reason is stated once rather than repeated across every column's verdict) naming WHY
     (frame-blind / inherited-from-refusal) — tying the marker to a reason, not a bare label. This
     **promotes the previously-voluntary rationale to mandatory** for structural boundary entries.

**Declarations (all five current boundary entries are `structural`):**

| entry | provenance | why |
|---|---|---|
| `spadl.add_restart_coordinates` | structural | reads no velocity-sensitive input; `identical` (frame-coupling tripwire) |
| `xtgk.compute_xt_gk_v2` | structural | frame-blind; `identical` |
| `gkdv.build_ghost_frames` | structural | inherited from the ADR-054 refusal; `honest_nan` |
| `gkdv.delta_das` | structural | inherited; `honest_nan` |
| `gkdv.delta_threat_suppression` | structural | inherited; `honest_nan` |

`spadl.add_restart_coordinates` is **already registered**, so this cycle adds its `verdict_provenance`
too — the gate's population is every *registered* boundary entry, and it would fail on a bare one.

**Anti-rot property (frame-blind half):** a NEW frame-blind boundary entry added as a bare
`identical` -> `works` fails the gate until it declares `structural` with a reason; a velocity-consuming
boundary entry that degrades cannot mislabel itself `structural` (the `differs`/`silent_degrade`
admissibility rule forces `substantive`). The gate derives its population from `BOUNDARY_ENTRY_POINTS`
and asserts over it exactly, so it cannot silently miss a new entry.

**Known limit — the gate locks HALF the distinction, and that half is the checkable one.** `works`
(from `identical`) forces `structural`, tight, because a value that cannot move across the velocity
legs was not substantively handled — this locks `xtgk.compute_xt_gk_v2`. `differs_by_design`/
`silent_degrade` forces `substantive`, enforceable but **inert this cycle** (no boundary entry observes
it). But **`honest_nan` is observationally ambiguous**: a function that refuses velocity-less input
ITSELF (substantive, the shape `add_ghost_gk` has) and one that INHERITS an upstream refusal
(structural, gkdv's shape) both produce `all_nan` -> `honest_nan`, and **no observation distinguishes
self-refusal from inherited-refusal**. So gkdv's `structural` is **author-asserted**: the gate forces a
`provenance_rationale` once `structural` is declared, but nothing forces the `structural` *choice* for
an `honest_nan` entry (and `RATIONALE_ALWAYS` does not include `honest_nan`). A future maintainer could
add a gkdv-shaped `honest_nan` boundary entry and mis-declare it either way; the gate would pass it.
Naming that ceiling IS the durability contribution — the enforcement is real for the frame-blind case
and rationale-documented-but-author-asserted for the inherited-refusal case, and this Part must not be
read as more than that.

## ADR-053 amendment

Amend the audit ADR with the boundary-entry policy this cycle establishes. The **load-bearing**
addition is a verdict-provenance distinction the amendment must make explicit, so an empty
`UNAUDITABLE_BOUNDARY` is never read as end-to-end degradation coverage:

- **Substantive vs structural/inherited verdicts.** A **substantive** degradation verdict comes from a
  velocity-*consuming* function whose own handling moves the value — it zero-fills (`differs_by_design`)
  or honest-NaNs (`honest_nan`) *because of what the function does*. A **structural/inherited** verdict
  comes from a function the audit's axes cannot substantively reach: **frame-blind ->
  `identical`/`works`** (the value cannot move — a frame-coupling tripwire), or
  **downstream-of-a-refusing-seam -> `honest_nan`** (the value is NaN because an upstream seam refused,
  not because this function handled anything). Both are honest recorded verdicts; **only the first is
  degradation coverage.** Registering a structural verdict retires the *un-auditable category* without
  claiming the function is velocity-audited. Both of this cycle's subjects are structural/inherited.
- **Frame-blind, injected-port orchestrators** are audited via *synthesize-and-inject*: deterministic,
  velocity-blind, live port doubles held identical across both legs (citing `audit_xt` and
  `visible_area_coverage`, which both resolve to `identical`/`works` — the same verdict, not two
  flavours). No bundled model is required, and the "no xG model" block was a **category error** (the
  audit measures degradation, not value quality). **`works` on a frame-blind orchestrator means "this
  function fabricates nothing through a frame it never reads" — NOT "the metric is velocity-robust or
  SB360-computable."** For `compute_xt_gk_v2` specifically, the metric's velocity-dependence lives in
  its inputs (`pressure`, `is_gk_distribution`, `retention_features`), computed upstream from tracking
  and unavailable on real SB360.
- **A function downstream of a refusing seam inherits that seam's verdict, and the inheritance is
  contingent.** gkdv inherits `serve_ghost_gk_positions`'s ADR-054 refusal -> `honest_nan` (the class
  `add_ghost_gk` already carries). The gkdv arms are **never reached** on the freeze-frame leg, so
  their *intrinsic* zero-velocity behaviour is **out of scope** and would need its own probe if the
  serving seam ever stops refusing (cf. ADR-063).
- **The distinction is enforced for the checkable half, author-asserted for the other.** A per-entry
  `verdict_provenance` (`substantive`/`structural`) field on every registered `BOUNDARY_ENTRY_POINT`,
  plus a meta-gate deriving its population from `BOUNDARY_ENTRY_POINTS`: `works` (from `identical`)
  forces `structural`, locking the frame-blind case (`xtgk`) tight; `differs_by_design`/
  `silent_degrade` forces `substantive`. But **`honest_nan` is observationally ambiguous** —
  self-refusal (substantive) and inherited-refusal (structural) produce the same `all_nan`, so the gate
  **cannot** check gkdv's `structural` choice; it is **author-asserted**, carrying a mandatory
  `provenance_rationale` but not a forced token. See Part 4's Known limit. So an empty
  `UNAUDITABLE_BOUNDARY` is safe from being re-read as end-to-end coverage **for the frame-blind half**
  and rationale-documented for the inherited-refusal half — not left to bare prose (ADR-056: a floor
  cannot detect an omission), but not fully machine-locked either, and that ceiling is named rather
  than papered over.
- Delete the stale TODO "four boundary entry points are unauditable" item as **resolved** (not
  tracked forward).

## Non-vacuity and correctness guards

- **xtgk liveness:** the five columns must be finite and non-constant on the scored rows (the port
  doubles are LIVE), so `identical` is a real number comparison, never `NaN==NaN`. A guard asserts the
  doubles move every one of the four decomposition terms. This certifies the *comparison* is real; it
  does **not** certify velocity handling — there is none to certify (C1). The `identical` verdict's
  value is as a frame-coupling regression tripwire, nothing more.
- **xtgk non-masking:** the doubles are **velocity-blind by construction** (they read zones/pressure
  derived from `actions`, never frames), so they cannot hide a velocity dependence — and the function
  has no frame parameter through which one could enter.
- **gkdv counterfactual non-vacuity:** the two arms must differ measurably between the factual and
  ghost legs on Leg B (the existing `tests/gkdv/test_arms.py` non-vacuity gates already pin this);
  the audit adapter must not collapse them by sharing a `PitchControlCache` (the arms already forbid
  it — `_arms.py:7`).
- **Observations are transcribed from execution**, never guessed: the predicted tables above are
  hypotheses; the recorded `AxisVerdict.observation` is whatever `run_axis` produces, and CI
  re-derives and locks it. The human adjudication + rationale is written against the observed value.

## Explicitly not in scope

- **No xG model, no bundled GhostGkModel.** Both are avoided by design; a future value-quality audit
  of `MarkovPossessionValue` is a separate concern the SB360 degradation audit does not cover.
- **No change to shared-fixture player positions.** gkdv possession is resolved by an adapter-supplied
  `carrier`; the fixture's positions (and therefore every other locked verdict) are untouched.
- **The gkdv arms' internal zero-velocity behaviour is not separately probed here** — the ghost-serving
  refusal on Leg A dominates (zero scored frames), so the arms are never reached on the freeze-frame
  leg. That is the honest thing to record.

## Caller / consumer sweep (changed public surface)

No public *library* signature changes — this cycle is test-registry + docs. Changed *test* seams:
`tests/sb360/_entries/_boundary.py` (4 new entries, each with its `call` **inline** exactly as
`_call_restart_coordinates` already is — boundary entries are hand-maintained and are NOT part of
`scripts/_sb_battery.py`'s `ADAPTER_MAP`, which serves the `tracking.__all__` `add_*` surface the
regenerator loops; plus a `verdict_provenance="structural"` on all 5 boundary entries, including the
existing `spadl.add_restart_coordinates`); `tests/sb360/_vocabulary.py` (new `VERDICT_PROVENANCE`
token set); `tests/sb360/_registry.py` (new `Sb360Entry.verdict_provenance` field;
`NOT_EXERCISED_BUDGET`); `tests/sb360/test_registry_surface.py` (`UNAUDITABLE_BOUNDARY` emptied, xfail
removed, new `test_boundary_entries_declare_admissible_provenance` gate). The gkdv threat adapter imports the
shared `audit_xt` from `scripts/_sb_battery.py` (the existing `tests -> scripts` layering that
`tests/sb360/_calls.py` already uses); it adds no new library dependency.

**To confirm during implementation:** whether `scripts/validate_sb360_licensed_corpus.py` iterates
the boundary entry points at all. It runs the `add_*` surface via `ADAPTER_MAP`, so it almost
certainly does **not** touch boundary entries — matching how `spadl.add_restart_coordinates` (a
boundary entry since before this cycle) is already treated. If that holds, the licensed-corpus driver
needs no change; if it does iterate boundary entries, the inline adapters must be import-reachable from
`scripts/` without forking the call convention.

## Testing strategy

Per entry: write the adapter, run `run_axis` to **observe** the machine observation, transcribe it
into the entry, write the human adjudication + rationale, and let the CI re-derivation lock it.
Red-first is expressed through the registry meta-gates: the completeness assertion is what flips from
xfail to passing once all four are registered, and `test_every_visibility_roster_has_its_own_slot` /
`test_every_verdict_is_admissible_from_its_observation` fail loudly if a roster block or an
inadmissible verdict is missing.
