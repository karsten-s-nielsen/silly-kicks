# ADR-053: SB360 coverage audit — observation/adjudication split

**Status:** Accepted
**Date:** 2026-08-04
**Spec:** `docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md`
**Supersedes:** the two-state compatibility table in
`docs/superpowers/specs/2026-05-27-snapshot-to-tracking-frames-design.md`

## Context

`tracking.snapshot_to_tracking_frames` has converted StatsBomb 360 freeze-frames into the
tracking schema since 4.x, and its spec documented which downstream functions worked on the
result. That table used two states — *works* and *gracefully degrades* — covered ~10 functions,
and had no gate. `tracking.__all__` subsequently grew to 33 `add_*`; `gkdv/` and `xtgk/` landed.
Nothing failed when the surface grew, because nothing was checking.

A commercial SB360 goalkeeper collaboration made the question load-bearing: which metrics
already work on freeze-frames, and which only appear to?

## Decision

**1. Five states, not two.** The two-state vocabulary cannot express the dangerous case: a
function that returns finite, plausible numbers computed from absent kinematics. Downstream that
is indistinguishable from a working feature. The audit adds `silent_degrade` alongside `works`,
`honest_nan`, `differs_by_design` and `not_exercised`.

**2. Separate the machine OBSERVATION from the human ADJUDICATION, and lock only the first.**
This is the load-bearing decision.

* The **observation** (`identical` / `differs` / `all_nan` / `partial_nan` / `no_signal` /
  `raises_a`) is re-derived by execution on every CI run and asserted against the registry.
* The **adjudication** (`works` / `silent_degrade` / `differs_by_design` / `honest_nan` /
  `not_exercised` / `raises`) is written by a human and carries a mandatory rationale.

Repair a function and its observation changes, CI fails, and the adjudication is forced to be
revisited.

**Rejected: locking the verdict itself.** The first design pinned registry KEYS to
`tracking.__all__` in both directions — a new export without a verdict failed CI. But nothing
pinned the verdict VALUE to observed behaviour, so repairing a function would leave a stale
`silent_degrade` in place while CI stayed green. That is a rot-resistant index over rotting
content: the audit's own defect class, relocated one layer up.

**Rejected: locking the adjudication too.** A machine cannot distinguish *fabricated* from
*legitimately different*. Pitch control evaluated at zero velocity is a well-defined
**positional** model — weaker, not invented. A fitted model silently imputing features it was
trained on is fabrication. Both observe `differs`. Only a human reading the feature can say
which, and the written rationale is the reviewable artifact.

That this prong is load-bearing was demonstrated the hard way, on this very example. The
`add_ghost_gk` rationale first shipped saying the model "receives structural zeros"; measured
later, it does not. `extract_ghost_gk_features` yields **NaN**, and `predict_mean`'s HGBR
reconstruction routes NaN down each split's *learned missing-value direction* — a different
prediction from zero-fill (`NaN → [6.795, 33.522]` vs `zero → [6.888, 33.362]`). The verdict was
right and the reasoning was wrong, which is precisely the failure a locked machine observation
cannot catch and a reviewable human rationale can. Corrected in 4.75.0 at the generating rule
(`tests/sb360/_adjudicate.py`), not at the 4 generated call sites.

**3. Verdicts are per emitted COLUMN, and per axis.** `add_action_context` splits — three
positional columns work, `actor_speed` is `all_nan`. Two independent axes are swept: velocity
(roster fixed) and visibility (kinematics fixed, roster ablated), each varying ONE factor, since
varying both makes a verdict unattributable.

**4. `differs` and `all_nan` are each reachable two ways, so the cause is isolated.** A
diagnostic third leg (anchor-only: one frame, WITH velocity) separates *velocity* from *frame
count*. Without it, a feature that merely needs a temporal window is indistinguishable from one
fabricating numbers from absent kinematics — and only the second is a finding.

**5. Coverage is measured on the ACTION side as well as the frame side.** Per-frame metrics are
structurally blind to an action that received no frame at all. For goal kicks that is the whole
story.

## Consequences

Tests, scripts and docs only. No library change, no aggregator added, C4-free, no retrain
trigger. A new `add_*` must carry an SB360 verdict or CI fails.

**Findings.** 299 of 486 verdicts are `works`; 4 are `silent_degrade`, all `add_ghost_gk`. The
xT-GK v1 surface (16 columns) works on freeze-frames and does not require the keeper to be in
frame. For the GK domain the binding constraint is neither the code nor keeper visibility but
whether a freeze-frame **exists** for the event being valued: goal kicks carry one ~23% of the
time (0–50% by match, 9 open matches).

**A library finding, recorded not fixed:** `snapshot_to_tracking_frames`' id dtype is
pandas-version-dependent — `Int64` in yields `Int64` on pandas 2.3.3 and `Float64` on 3.0.3.

**Four boundary entry points are out of scope**, enumerated with per-name reasons behind a
strict xfail. `xtgk.compute_xt_gk_v2` needs an xG-calibrated port and silly-kicks ships no xG
model, so any port supplied would audit the stub rather than the library.

## What this cost, and why the process is part of the decision

The design took five review rounds, and **each of the first four introduced a version of the
same defect while repairing the previous one**: a state defined at one level and absent from a
table claiming completeness. That is why the vocabulary is declared once, namespaced, and
carries its own completeness gate — reviewing again finds the next instance; a mechanical check
stops it.

Execution then overturned four claims that had survived all five reviews, including the audit's
own motivating example (`add_gk_influence` was believed to fabricate; it declines). **A reachable
code path is not evidence about the value it produces** — the zero-fill it was accused of is real
and IS reached, and the output is still NaN.

## Amendment (2026-08-20, silly-kicks 4.88.0, PR-S158) — boundary-audit closeout: verdict provenance, `UNAUDITABLE_BOUNDARY` emptied

The four boundary entry points parked as out-of-scope behind a strict xfail
(`xtgk.compute_xt_gk_v2`, `gkdv.build_ghost_frames`, `gkdv.delta_das`,
`gkdv.delta_threat_suppression`) are now **registered**, and `UNAUDITABLE_BOUNDARY` is **empty**.
The `xtgk.compute_xt_gk_v2` reason -- "needs an xG-calibrated `MarkovPossessionValue` port and
silly-kicks ships no xG model, so any port audits the stub" -- was a **category error**: the audit
measures **velocity/frame degradation** (a cross-leg delta), never **value quality**, so a
deterministic port double injected identically into both legs cannot audit the stub (its values
cancel in the delta). The load-bearing addition is a verdict-provenance distinction the amendment
makes explicit, so an empty `UNAUDITABLE_BOUNDARY` is never re-read as end-to-end degradation
coverage.

- **Substantive vs structural/inherited verdicts.** A **substantive** degradation verdict comes
  from a velocity-*consuming* function whose own handling moves the value -- it zero-fills
  (`differs_by_design`) or honest-NaNs (`honest_nan`) *because of what the function does*. A
  **structural/inherited** verdict comes from a function the audit's axes cannot substantively
  reach: **frame-blind -> `identical`/`works`** (the value cannot move -- a frame-coupling
  tripwire), or **downstream-of-a-refusing-seam -> `honest_nan`** (the value is NaN because an
  upstream seam refused, not because this function handled anything). Both are honest recorded
  verdicts; **only the first is degradation coverage.** Registering a structural verdict retires
  the *un-auditable category* without claiming the function is velocity-audited. Both of this
  cycle's subjects are structural/inherited.
- **Frame-blind, injected-port orchestrators** are audited via *synthesize-and-inject*:
  deterministic, velocity-blind, live port doubles held identical across both legs -- the same
  pattern `audit_xt` (`scripts/_sb_battery.py`; injects a non-degenerate `ExpectedThreat` into six
  aggregators) and `visible_area_coverage` (a frame-blind entry that records the honest "the axis
  cannot reach it" rather than manufacturing a difference) already use; both resolve to
  `identical`/`works` -- the same verdict, not two flavours. No bundled model is required.
  **`works` on a frame-blind orchestrator means "this function fabricates nothing through a frame
  it never reads" -- NOT "the metric is velocity-robust or SB360-computable."** For
  `compute_xt_gk_v2` specifically, the metric's velocity-dependence lives in its inputs
  (`pressure`, `is_gk_distribution`, `retention_features`), computed upstream from tracking and
  unavailable on real SB360; the `identical` verdict is a **regression invariant** that flips if a
  future edit makes the function read a frame, nothing more.
- **A function downstream of a refusing seam inherits that seam's verdict, and the inheritance is
  contingent.** gkdv inherits `serve_ghost_gk_positions`'s ADR-054 refusal of velocity-less
  freeze-frames -> `honest_nan` (the class `add_ghost_gk` already carries): on the freeze-frame
  leg no ghost is served, zero frames are scored, and the arms are **never reached**. So the arms'
  *intrinsic* zero-velocity behaviour is **out of scope** and would need its own probe if the
  serving seam ever stops refusing (cf. ADR-063, which lifted the four pitch-control aggregators to
  the zero-velocity positional model).
- **The distinction is enforced for the checkable half, author-asserted for the other.** A
  per-entry `verdict_provenance` (`substantive`/`structural`) field on every registered
  `BOUNDARY_ENTRY_POINT` (paired with a mandatory `provenance_rationale`), plus a meta-gate
  (`test_boundary_entries_declare_admissible_provenance`) deriving its population from
  `BOUNDARY_ENTRY_POINTS`: `works` (from `identical`) forces `structural`, locking the frame-blind
  case (`xtgk.compute_xt_gk_v2`) tight -- a value that cannot move across the velocity legs was not
  substantively handled; `differs_by_design`/`silent_degrade` forces `substantive`. **Known limit:
  `honest_nan` is observationally ambiguous** -- self-refusal (substantive, the shape
  `add_ghost_gk` has) and inherited-refusal (structural, gkdv's shape) both produce `all_nan`, and
  no observation distinguishes them, so the gate **cannot** check gkdv's `structural` choice; it is
  **author-asserted**, forced only to carry a rationale. This gate locks HALF the distinction, and
  naming that ceiling is the durability contribution -- it is not overstated as fully test-locked
  (ADR-056: a floor cannot detect an omission, so the frame-blind half is machine-checked; the
  inherited-refusal half is rationale-documented but author-asserted).

All five current boundary entries (the four above plus the already-registered
`spadl.add_restart_coordinates`) declare `verdict_provenance="structural"` with a stated reason.
The stale TODO "four boundary entry points are unauditable" item is deleted as **resolved** (not
tracked forward). Consequences unchanged from the parent decision: tests and docs only, no library
change, no aggregator added, C4-free, no retrain trigger.

### Follow-up (ADR-067 Phase B) — the gate refined for the first MIXED entry; two entries now `substantive`

Bundling the position-only ghost (ADR-067) made gkdv's counterfactual arms *produce* on SB360, so
`gkdv.build_ghost_frames` and `gkdv.delta_threat_suppression` became **`substantive`** (their values
now move across the velocity legs) — the "all five declare `structural`" statement above no longer
holds. `delta_threat_suppression` is also the first **MIXED** boundary entry: substantive where there
is threat, but a coincidental `0==0` `works` on the no-threat goalkick roster. The per-cell
`works→structural` lock could not express that mix, so the gate was refined: an entry is `substantive`
if ANY cell moves (`differs_by_design`/`silent_degrade`), exempting coincidental `works` cells; an
entry with NO substantive cell still locks `works→structural`, keeping the frame-blind case
(`add_restart_coordinates`, `xtgk.compute_xt_gk_v2`) tight. See ADR-067's Phase-B amendment.
