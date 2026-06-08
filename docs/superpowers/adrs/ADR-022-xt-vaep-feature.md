# ADR-022: xT as a VAEP feature (`xt__<method>` xfn factory)

| Field | Value |
|---|---|
| **Date** | 2026-06-08 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen (silly-kicks); cross-session reviewer (4 review rounds) |

## Context

SK-xT-1 (4.17.0, ADR-021) turned `xthreat` into a pluggable, evaluatable package, but xT remained a
*standalone model* and an input to several tracking features — never a per-action VAEP feature. The
SK-xT-1 spec explicitly deferred "a new VAEP `xt__<method>` xfn factory … can be added later". This
is that follow-up.

The forcing constraints: (1) a VAEP feature must be train/serve consistent — the value surface used
at training time must be byte-identical at serve time; (2) the standard `ExpectedThreat.rate()`
filters by standard-SPADL move types + `result_id==success`, but **atomic SPADL has no per-action
`result` column** (success is encoded in a *following* atom — `receival`), and dribbles are inserted
as same-team carries that are *never* followed by a `receival`; (3) `convert_to_atomic` renumbers
`action_id`, so `action_id` is not a stable cross-representation key.

## Decision

Ship `xt_xfns(*, model)` (standard + atomic) — a factory that closes over a **caller-supplied fitted
`ExpectedThreat`** and emits one frame-free `xt__<model.method>` column per gamestate slot, NaN for
non-move/failed-move actions, kept out of every default xfn list (opt-in). The atomic mirror reuses
the **exact `model.rate()` path** by synthesizing a standard-SPADL-shaped frame with a **type-aware**
`result_id` (dribble intrinsic; pass/cross success iff the next atom is `receival`) — `rate()` is
left untouched.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Inline-fit xT inside the xfn | zero caller setup | fits on the data it then rates (leak); non-reproducible at serve | violates train/serve consistency |
| B. Bundle a frozen default grid now | turn-key | full ADR-011 model-lifecycle weight (provenance, card, SHA256, retrain semantics) | separate deliverable; reserved via the typed `str` door |
| C. Add to `xfns_default` | xT available by default | forces a global retrain for every provider; imposes a non-standard methodological choice | opt-in instead |
| D. Extract a shared `_rate_cells` cell-lookup helper for the atomic path | shares the cell lookup | refactors shipped `rate()` (parity risk); shares only the lookup; cross-package private-symbol import | superseded by E |
| E. (chosen) atomic reuses `model.rate()` via a synthesized `result_id` | entire rating path shared (filter + NaN-drop + y-flip + delta); `rate()` untouched; public-only dependency | constructs one small adapter frame per game | — |

## Consequences

### Positive

- xT becomes a first-class, opt-in VAEP feature for both standard and atomic SPADL, column-symmetric
  (`xt__<method>` means the same thing in both representations).
- `rate()` is the single rating path; the standard `rate()` is literally unmodified, so the SK-xT-1
  byte-identical parity gate and golden snapshots remain trivially green.
- `ExpectedThreat` is imported only under `TYPE_CHECKING` (the model is duck-typed at runtime), so
  there is **no new runtime dependency edge** and bare `import silly_kicks` is unaffected.
- Additive: zero forced retrain. A guard test asserts the symbols are absent from all default lists.

### Negative

- A caller who opts the feature into their xfns triggers their *own* VAEP retrain (documented).
- The atomic adapter synthesizes a standard-SPADL frame; a maintainer must understand the
  type-aware-`result_id` translation (documented in code + tests).

### Neutral

- **Inherent representation edge:** a pass/cross that is the *last action of a period* has no
  following atom, so the atomic feature yields NaN where standard may rate it (≤1 action/period).
  Documented, not papered over.
- Boundary slots map by the composite `(game_id, period_id, action_id)` key (not bare `action_id`),
  resolving to the `gamestates()` first-in-group fill — symmetric with the standard `@simple` path,
  never forced to NaN.

## Related

- **Specs:** `docs/superpowers/specs/2026-06-08-xt-vaep-feature-design.md`
- **Plans:** `docs/superpowers/plans/2026-06-08-xt-vaep-feature.md`
- **ADRs:** builds on `ADR-021` (pluggable xT); follows `ADR-009` (FrozenXt exogenous-artifact
  discipline) and `ADR-005` §8 (`<feature>__<method>` naming) and `ADR-011` (the reserved bundled-grid
  lifecycle door).
- **External references:** Singh, K. (2018), "Introducing Expected Threat (xT)" — see NOTICE.
