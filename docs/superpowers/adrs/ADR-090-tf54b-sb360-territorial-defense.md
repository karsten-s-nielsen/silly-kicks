# ADR-090: TF-54b SB360 territorial-defense counterfactual (removal/marginal) + actor-identity bridge + kept expected-passing seams

| Field | Value |
|---|---|
| **Date** | 2026-09-09 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

TF-54 (ADR-086) shipped `silly_kicks.territory` — an event-only per-`(player, match)` valuation of the
ground a defender patrols (the "Van Dijk" metric) — with a **reserved typed `counterfactual` door**
(`TERRITORY_METHODS = {completed_failed, counterfactual}`, the second raising `NotImplementedError`).

TF-54b was originally scoped to implement that door **event-only**: a defender's "prevented" as an
expected−realized GSAA-analog over opponent passes aimed into the hull (the `ab9001c` work —
`territory/_counterfactual.py` + `expected_passing/` + `xthreat/_counterfactual_seam.py`). That work was
committed on branch `feat/tf54b-counterfactual-territorial-prevented` (`ab9001c`, drafted as
PR-S182/ADR-089) but **never merged** — the TF-60 Layer-3 cycle took the PR-S182/ADR-089 slot (the merged
4.111.0). The event-only approach carries an inherent limit: a failed pass's **intended target is
unobservable from events alone** (SPADL `end` is the death location), so the "prevented" leg is a
death-location proxy and the real-data mechanism leg is infeasible event-only.

SB360 freeze-frames give the missing spatial ground truth: the actual positions of the defender and their
teammates at the moment of the action. The owner re-scoped TF-54b to a **tracking-consuming** metric —
how much does a defender's *positioning* suppress the attacking team's threat, measured by a **model-free
removal counterfactual** (Fernández–Bornn marginal player value): remove the defender's row and let pitch
control re-partition the vacated space. SB360 freeze-frames are anonymous (`snapshot_to_tracking_frames`
numbers the rows; no player identity), so identity-exact attribution needs a bridge from the one row SB360
*does* implicitly identify — the actor.

## Decision

Ship a new tracking-consuming `silly_kicks.territorial_defense` package as one cycle (two
provenance-mandated commits — library code + version/docs now; owner-run bundled weights + construct-validity
battery post-commit, merged non-squash). The five owner rulings (locked 2026-09-09):

1. **Mechanism = removal/marginal, replacement deferred.** `a_threat_suppressed = compute_threat_pc(cf) −
   compute_threat_pc(actual)` where `cf` is the factual frame with **one defender row removed**; pitch
   control re-partitions the vacated space to the remaining players. Replacement (a ghost keeper/defender,
   "above-replacement") is a future refinement.
2. **Package = new `territorial_defense/`** — a hexagonal sibling of `gkdv/` and `restdefense/`: imports
   `silly_kicks.tracking` public seams only; **nothing imports it** (AST allowlist gate). A `compute_*`,
   NOT an `add_*` (the 33 action-coupled aggregator count is unchanged); in NO default xfn list.
3. **Event-only counterfactual NOT carried forward; reusable kept seams retained.** The `ab9001c`
   event-only *cone* (`territory/_counterfactual.py`, `_synthetic_interception.py`,
   `validate_territory_counterfactual.py`, the cone tests + 4 cone glossary columns) is dropped. The
   independently-useful **kept seams** are retained as infrastructure: `expected_passing/` (the
   `PassCompletionModel`), `xthreat/_counterfactual_seam.py` (+ the `destination_profiles` export +
   `_transitions.py` change), `scripts/train_pass_completion.py`, `scripts/_sb_open_data.py`.
4. **Two arms.** **Arm A** (`a_threat_suppressed`) — action-anchored, **identity-EXACT** via the actor
   bridge, over the acting defender's own defensive actions (tackle/interception/clearance,
   `type_id ∈ {9, 10, 18}`). **Arm B** (`b_threat_suppressed`) — hull-based, attribution-**approximate**,
   nearest-to-target defender removed (receiver-lane evaluated in the battery); reports
   `b_attribution_slippage` (the rate the contesting defender is NOT the hull-owner D — attribution
   error, lower = tighter; honest-NaN, never a fabricated 0, when identity is un-measurable).
5. **Vehicle = fresh commit off `main`, renumbered** (4.112.0 / PR-S183 / ADR-090), close draft PR #235.

Supporting decisions: `is_actor` is re-plumbed through the SB360 port as a **snapshot-only** extension
column (base `TRACKING_FRAMES_COLUMNS` unchanged); the actor bridge `apply_actor_identities_to_frames`
lives in the shared `silly_kicks/keeper_identity.py` (next to `apply_keeper_identities_to_frames`; ADR-078
bridge pattern, ADR-084 module home); `territory`'s reserved `counterfactual` door is **removed**
(`TERRITORY_METHODS → {completed_failed}`), since the counterfactual now lives in `territorial_defense`.

**Sign convention:** attacker-value units, **positive = threat suppressed** (`threat_pc(cf) −
threat_pc(actual)`) — INVERTED from gkdv's "negative = deterrent"; the probe's direction registry encodes
`positive`.

**HONEST LIMIT (load-bearing, stated in the CLAUDE.md contract, the module docstrings, the glossary
`definition`s, and the research report):** the metric is validated as an **INSTRUMENT, NOT as
player-attributable**. The marginal-removal delta is **team-conditioned by construction** (the vacated
space re-partitions to the defender's teammates). The validation corpus (WC2022 + 30 licensed SB360 —
national-team / single-tournament) **structurally cannot identify** the defender-vs-team confound: one
player = one national team → **zero cross-team defender observations**, so per-defender numbers are NOT a
ranking and the elite-defender ("Van Dijk") prior is elite-defender/elite-team **collinear** — a clean
prior is FACE-VALIDITY, not attribution evidence. Ranking is a future ADR-009 gated on a crossed
defender+team variance decomposition (ICC — not CV) over a **multi-club transfer** corpus. Precedent: the
sister GK-distribution metric (eyestone collaboration) passed its own face-validity checks then proved
~80% team-confounded ("ranking not licensed").

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Implement the event-only `counterfactual` door (`ab9001c`) | No tracking dependency; reuses `territory` hull | A failed pass's intended target is unobservable event-only; the "prevented" leg is a death-location proxy, and the real-data mechanism leg is infeasible | SB360 tracking gives the actual spatial counterfactual the event-only approach can only proxy |
| B. Replacement (ghost keeper/defender) counterfactual | Above-replacement framing; consistent with gkdv | Needs a trained ghost model; more moving parts | Removal is **model-free** (Fernández–Bornn marginal value); replacement is a future refinement |
| C. Add `counterfactual` as a `method` inside `territory/` | Single package | `territory` is event-only (imports `xthreat`, never `tracking`); a tracking counterfactual is a different dependency layer | Layering — a tracking-consuming counterfactual is a separate package; the reserved door is removed rather than left as dead `NotImplementedError` |
| D. New `territorial_defense/` + removal mechanism + actor bridge + kept seams (**chosen**) | Spatial, identity-exact (Arm A); model-free; kept seams landed as infrastructure | Not player-attributable (team-confound); Arm A narrow; PassCompletionModel weights owner-run | — |

## Consequences

### Positive

- A spatial, identity-exact (Arm A) territorial-defense **instrument** on SB360 freeze-frames.
- The actor bridge `apply_actor_identities_to_frames` is **independently useful** (the eyestone GK
  build-up-decision metric is a concrete second consumer) — hence its shared `keeper_identity.py` home.
- `expected_passing.PassCompletionModel` + `xthreat.destination_profiles` land as **reusable
  infrastructure** rather than being lost with the unmerged `ab9001c`.
- Fixed a **real ADR-055 crash**: `compute_threat_pc` raised `GoalEndUnresolvedError` on keeperless
  freeze-frames; the arms now catch it at the edge and degrade to honest-NaN (`unresolved_geometry`),
  surfaced during the SB360 boundary-audit fixture extension.

### Negative

- **Not player-attributable** (team-confound, above) — per-defender numbers are not a ranking.
- Arm A is **narrow** (identity-exact only on the actor's own defensive actions); Arm B carries
  **attribution slippage** (the contesting defender may not be D — reported as `b_attribution_slippage`).
- `PassCompletionModel.bundled()` raises `FileNotFoundError` until the owner-run weights land (commit 2);
  two `bundled()` tests skip until then.
- The metric is **reported-not-gated** — the construct-validity battery is owner-run (DGX/corpus) and
  promotes NO default (any promotion is a separate ADR-009 decision).

### Neutral

- **+2 C4 containers** (`territorial_defense` **and** `expected_passing` — the latter is new-to-`main`,
  carried from the unmerged `ab9001c`, so the C4 subpackage-completeness gate requires its container too;
  the plan's "+1" pre-dated that observation); **+3 feature-glossary columns**
  (`a_threat_suppressed`/`b_threat_suppressed` = xT, `b_attribution_slippage` = ratio).
- **No VAEP/tracking retrain, no re-materialize** — additive; every existing feature column byte-identical;
  the new package is in NO default xfn list; `compute_*` not `add_*` (33-count unchanged).
- `pitch_control_method="spearman"` is a **hard `__post_init__` constraint** (a GK-blind method is
  unrepresentable, mirroring `GkdvParams`); the arms accept **no `pitch_control_cache`** (ADR-043
  landmine: the identity-keyed cache would serve the factual surface to the counterfactual and collapse
  every delta to exactly 0).

## CLAUDE.md Amendment

None required. This ADR is additive: it adds a `territorial_defense` durable-contract bullet (carrying the
honest-limit verbatim in spirit), updates the `territory` bullet to reflect the removed `counterfactual`
door, and adds a note that `expected_passing` / `xthreat.destination_profiles` are landed as reusable
seams. No project-wide rule is excepted.

## Related

- **Specs:** `docs/superpowers/specs/2026-09-06-tf54b-sb360-ghosting-territorial-defense-design.md`
- **Plans:** `docs/superpowers/plans/2026-09-09-tf54b-sb360-territorial-defense.md`
- **Issues / PRs:** closes draft `#235`; opens a new PR from `feat/tf54b-sb360-territorial-defense`
- **ADRs:** builds on ADR-086 (`territory`), ADR-084/ADR-078 (`keeper_identity` bridge pattern + module
  home), ADR-043 (gkdv `PitchControlCache` landmine + the arms-in-attacker-value-units idiom), ADR-055
  (`resolve_defended_goals` / honest-NaN at the edge), ADR-063 (velocity tiers), ADR-077 (FOV honest-NaN),
  ADR-042 (dropped-and-counted conservation), ADR-052/ADR-037/ADR-056 (driver shards / provenance / input
  contract). Supersedes the unmerged `ab9001c` event-only counterfactual approach (the reusable seams are
  retained; the cone is not).
- **External references:** Fernández & Bornn (pitch control / marginal player value), Spearman (pitch
  control), Le et al. 2017 (ghosting comparator, not implemented).
