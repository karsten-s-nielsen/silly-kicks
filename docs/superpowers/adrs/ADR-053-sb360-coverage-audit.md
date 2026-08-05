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
**positional** model — weaker, not invented. A fitted model fed structural zeros for features it
was trained on is fabrication. Both observe `differs`. Only a human reading the feature can say
which, and the written rationale is the reviewable artifact.

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
